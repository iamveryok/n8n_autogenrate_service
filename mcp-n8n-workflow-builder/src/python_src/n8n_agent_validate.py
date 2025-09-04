import asyncio
import os
import json
import re

import aiohttp
import json5
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from jsonschema import validate, ValidationError
from dataclasses import dataclass
from datetime import datetime
import hashlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


@dataclass
class ErrorPattern:
    error_pattern: str
    solution: str
    priority: int = 1


class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []

    def __bool__(self):
        return self.is_valid and not self.errors


class AgentState(BaseModel):
    user_input: str
    workflow_json: str
    deployment_result: Dict
    error_messages: List[str]
    iteration_count: int
    max_iterations: int
    validation_results: List[ValidationResult]
    current_stage: str


class N8NWorkflowAgent:
    def __init__(self):
        self.llm = BaseChatOpenAI(
            model="deepseek-chat",
            openai_api_key='sk-d55e63ecb0534036972a3fe3e1370687',  # 替换为你的API密钥
            openai_api_base='https://api.deepseek.com',
            temperature=0.1
        )

        # n8n API配置
        self.n8n_base_url = os.getenv("N8N_BASE_URL", "http://localhost:5678")
        self.n8n_api_key = os.getenv("N8N_API_KEY")

        # 错误模式库
        self.error_patterns = self._initialize_error_patterns()

        # 验证缓存
        self.validation_cache: Dict[str, ValidationResult] = {}

        # 系统提示词
        self.system_prompt = self._create_system_prompt()

        # n8n JSON Schema
        self.n8n_schema = self._create_n8n_schema()

        # 加载节点配置文件
        print("正在加载node_config.json文件...")
        self.base_nodes = self.load_base_nodes()
        print(f"成功加载 {len(self.base_nodes)} 个节点规范")

        # 🔵 关键：禁用SSL（n8n本地部署默认用HTTP，无需加密）
        self.connector = aiohttp.TCPConnector(
            limit=10,  # 控制最大连接数
            enable_cleanup_closed=True,  # 自动清理关闭的连接
            ssl=False  # 明确禁用SSL，解决配置冲突
        )
        # 创建全局会话（复用连接池）
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=30),
            connector_owner=False  # 会话不拥有连接池，避免提前关闭
        )


    def load_base_nodes(self) -> Dict:
        """加载node_config.json文件"""
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            base_node_path = os.path.join(base_dir, 'nodes_config.json')
            print(f"  读取node_config.json文件: {base_node_path}")
            with open(base_node_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            nodes = data.get('nodes', [])
            print(f"  解析到 {len(nodes)} 个节点定义")
            # 构建节点类型映射
            node_dict = {node['type']: node for node in nodes}
            print(f"  构建节点类型映射，共 {len(node_dict)} 个唯一节点类型")
            # 显示一些节点类型示例
            sample_types = list(node_dict.keys())[:5]
            print(f"  节点类型示例: {sample_types}")
            return node_dict
        except FileNotFoundError:
            print("找不到node_config.json文件")
            return {}
        except json.JSONDecodeError as e:
            print(f"node_config.json文件格式错误: {e}")
            return {}
        except Exception as e:
            print(f"加载node_config.json失败: {e}")
            return {}

    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """初始化错误模式库"""
        return [
            ErrorPattern(
                error_pattern=r"JSON.*(invalid|parse|syntax)",
                solution="修复JSON语法错误，确保引号、括号匹配，逗号使用正确",
                priority=1
            ),
            ErrorPattern(
                error_pattern=r"node.*not.*found|unknown.*node",
                solution="检查节点类型是否正确，使用有效的n8n节点类型如'n8n-nodes-base.httpRequest'",
                priority=1
            ),
            ErrorPattern(
                error_pattern=r"connection.*invalid|missing.*connection",
                solution="确保connections字段正确连接节点，检查节点ID引用是否正确",
                priority=2
            ),
            ErrorPattern(
                error_pattern=r"position.*required|missing.*position",
                solution="为所有节点添加position字段，格式为[x, y]",
                priority=2
            ),
            ErrorPattern(
                error_pattern=r"typeVersion.*required",
                solution="为所有节点添加typeVersion字段，通常为1或1.2",
                priority=2
            ),
            ErrorPattern(
                error_pattern=r"credential.*missing|authentication",
                solution="为需要认证的节点添加credentials字段，或确保认证配置正确",
                priority=3
            ),
            ErrorPattern(
                error_pattern=r"parameter.*missing|invalid.*parameter",
                solution="检查节点参数是否完整，确保必需参数都存在",
                priority=2
            )
        ]

    def _create_system_prompt(self) -> str:
        """创建系统提示词"""
        return f"""你是一个n8n工作流专家，能够根据用户需求生成符合n8n规范的JSON工作流。

当前时间: {datetime.now().isoformat()}

n8n工作流规范要求：
1. 必须包含name、nodes、connections、settings、versionId等基本字段
2. nodes数组中每个节点必须有id、name、type、typeVersion、position、parameters
3. connections必须正确连接各个节点，节点之间需要建立正确的连接关系
4. 所有节点type必须使用正确的n8n节点类型
5. 确保JSON格式完全正确，使用双引号
6. position格式为[x, y]数组
7. typeVersion通常为1或1.2

常见节点类型：
- n8n-nodes-base.scheduleTrigger (定时触发器)
- n8n-nodes-base.httpRequest (HTTP请求)
- n8n-nodes-base.code (代码节点)
- n8n-nodes-base.function (函数节点)
- n8n-nodes-base.emailSend (邮件发送)

错误模式库知识：
{self._get_error_patterns_knowledge()}

请根据用户需求生成完整的工作流JSON。确保语法正确且符合n8n规范。"""

    def _get_error_patterns_knowledge(self) -> str:
        """获取错误模式库知识"""
        knowledge = []
        for pattern in sorted(self.error_patterns, key=lambda x: x.priority, reverse=True):
            knowledge.append(f"- 错误模式: {pattern.error_pattern}")
            knowledge.append(f"  解决方案: {pattern.solution}")
        return "\n".join(knowledge)

    def _create_n8n_schema(self) -> Dict:
        """创建n8n JSON Schema用于验证"""
        return {
            "type": "object",
            "required": ["name", "nodes", "connections"],
            "properties": {
                "name": {"type": "string"},
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "name", "type", "typeVersion", "position", "parameters"],
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "typeVersion": {"type": ["number", "string"]},
                            "position": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "parameters": {"type": "object"}
                        }
                    }
                },
                "connections": {"type": "object"},
                "settings": {"type": "object"},
                "versionId": {"type": "string"},
                "active": {"type": "boolean"}
            }
        }

    def _get_workflow_hash(self, workflow_json: str) -> str:
        """获取工作流哈希值用于缓存"""
        return hashlib.md5(workflow_json.encode()).hexdigest()

    def validate_syntax(self, workflow_json: str) -> ValidationResult:
        """语法验证：JSON格式检查"""
        workflow_hash = self._get_workflow_hash(workflow_json)

        if workflow_hash in self.validation_cache:
            return self.validation_cache[workflow_hash]

        errors = []
        warnings = []

        try:
            # 尝试解析JSON
            data = json5.loads(workflow_json)

            # 检查基本结构
            if not isinstance(data, dict):
                errors.append("工作流必须是JSON对象")

            # 检查必需字段
            required_fields = ["name", "nodes", "connections"]
            for field in required_fields:
                if field not in data:
                    errors.append(f"缺少必需字段: {field}")

            result = ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )
            self.validation_cache[workflow_hash] = result
            return result

        except json.JSONDecodeError as e:
            error_msg = f"JSON语法错误: {str(e)}"
            result = ValidationResult(is_valid=False, errors=[error_msg])
            self.validation_cache[workflow_hash] = result
            return result

    def validate_schema(self, workflow_json: str) -> ValidationResult:
        """模式验证：n8n Schema检查"""
        workflow_hash = self._get_workflow_hash(workflow_json)

        if workflow_hash in self.validation_cache and self.validation_cache[workflow_hash].errors:
            return self.validation_cache[workflow_hash]

        errors = []
        warnings = []

        try:
            data = json5.loads(workflow_json)

            # 使用jsonschema验证
            validate(instance=data, schema=self.n8n_schema)

            # 额外自定义验证
            self._custom_validation(data, errors, warnings)

            result = ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )
            self.validation_cache[workflow_hash] = result
            return result

        except ValidationError as e:
            errors.append(f"Schema验证错误: {e.message}")
            result = ValidationResult(is_valid=False, errors=errors)
            self.validation_cache[workflow_hash] = result
            return result
        except Exception as e:
            errors.append(f"验证过程错误: {str(e)}")
            result = ValidationResult(is_valid=False, errors=errors)
            self.validation_cache[workflow_hash] = result
            return result

    def _custom_validation(self, data: Dict, errors: List[str], warnings: List[str]):
        """自定义验证规则"""
        # 检查节点ID唯一性
        node_ids = set()
        for node in data.get('nodes', []):
            node_id = node.get('id')
            if node_id:
                if node_id in node_ids:
                    errors.append(f"重复的节点ID: {node_id}")
                node_ids.add(node_id)

        # 检查连接引用
        connections = data.get('connections', {})
        all_node_names = {node.get('name') for node in data.get('nodes', []) if node.get('name')}

        for source_node, connection_info in connections.items():
            if source_node not in all_node_names:
                warnings.append(f"连接引用了不存在的源节点: {source_node}")

            for connection_list in connection_info.values():
                for connection in connection_list:
                    for link in connection:
                        target_node = link.get('node')
                        if target_node and target_node not in all_node_names:
                            warnings.append(f"连接引用了不存在的目标节点: {target_node}")

    def validate_semantics(self, workflow_json: str, user_input: str) -> ValidationResult:
        """语义验证：检查是否满足用户需求"""
        # todo 后续扩展
        errors = []
        warnings = []

        try:
            data = json5.loads(workflow_json)

            # 检查是否包含关键节点类型
            node_types = {node.get('type', '') for node in data.get('nodes', [])}

            # 根据用户输入检查关键功能
            user_input_lower = user_input.lower()

            if any(keyword in user_input_lower for keyword in ['定时', '每天', 'schedule', 'cron']):
                if not any('schedule' in node_type.lower() for node_type in node_types):
                    warnings.append("用户需求包含定时功能，但未找到定时触发器节点")

            if any(keyword in user_input_lower for keyword in ['http', 'api', '请求']):
                if not any('http' in node_type.lower() for node_type in node_types):
                    warnings.append("用户需求包含HTTP请求，但未找到HTTP请求节点")

            if any(keyword in user_input_lower for keyword in ['邮件', 'email', '发送']):
                if not any('email' in node_type.lower() for node_type in node_types):
                    warnings.append("用户需求包含邮件功能，但未找到邮件发送节点")

            return  ValidationResult(is_valid=True, errors=errors, warnings=warnings)

        except Exception as e:
            return ValidationResult(is_valid=False, errors=[f"语义验证错误: {str(e)}"])

    def match_error_pattern(self, error_message: str) -> Optional[ErrorPattern]:
        """匹配错误模式"""
        for pattern in sorted(self.error_patterns, key=lambda x: x.priority, reverse=True):
            if re.search(pattern.error_pattern, error_message, re.IGNORECASE):
                return pattern
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_workflow(self, user_input: str, error_history: List[str] = None) -> str:
        """生成n8n工作流JSON"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"用户需求: {user_input}")
        ]

        print(f"messages: {messages}")

        if error_history:
            error_context = "\n".join(
                [f"第{i + 1}次错误: {error}" for i, error in enumerate(error_history[-5:])])  # 只保留最近3个错误
            messages.append(HumanMessage(content=f"之前的错误信息:\n{error_context}\n请根据这些错误修正工作流。"))

        # 添加错误模式建议
        if error_history:
            last_error = error_history[-1]
            matched_pattern = self.match_error_pattern(last_error)
            if matched_pattern:
                messages.append(HumanMessage(content=f"错误模式建议: {matched_pattern.solution}"))

        response = await self.llm.ainvoke(messages)
        print(f"response1111: {response}")
        try:
            workflow_json = self._extract_json_from_response(response.content)
            # 多阶段验证
            syntax_result = self.validate_syntax(workflow_json)
            if not syntax_result:
                raise ValueError(f"语法验证失败: {', '.join(syntax_result.errors)}")

            schema_result = self.validate_schema(workflow_json)
            if not schema_result:
                raise ValueError(f"模式验证失败: {', '.join(schema_result.errors)}")

            semantic_result = self.validate_semantics(workflow_json, user_input)
            if semantic_result.warnings:
                logger.warning(f"语义警告: {', '.join(semantic_result.warnings)}")

            return workflow_json

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"生成的工作流验证失败: {e}")
            # 如果验证失败，让LLM重新生成
            return await self._regenerate_with_validation_feedback(user_input, str(e), error_history)



    def _extract_json_from_response(self, text: str) -> str:
        """从LLM响应中提取JSON内容"""
        # 查找JSON开始和结束的位置
        start = text.find('{')
        end = text.rfind('}') + 1

        if start != -1 and end != -1:
            json_str = text[start:end]
            # 清理JSON字符串
            json_str = re.sub(r',\s*]', ']', json_str)  # 修复尾随逗号
            json_str = re.sub(r',\s*}', '}', json_str)
            return json_str
        return text

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def _regenerate_with_validation_feedback(self, user_input: str, error_message: str,
                                             error_history: List[str]) -> str:
        """根据验证反馈重新生成"""
        feedback_prompt = f"""
之前的用户需求: {user_input}

生成的工作流验证失败，错误信息:
{error_message}

之前的错误历史:
{chr(10).join(error_history[-2:] if error_history else [])}

请根据验证错误重新生成正确的工作流JSON。确保:
1. 修复所有JSON语法错误
2. 确保符合n8n schema规范
3. 检查所有必需字段都存在
4. 确保节点类型正确
5. 确保节点之间建立正确的连接关系

只返回修正后的JSON，不要有其他文本。
"""

        response = await self.llm.ainvoke([
            SystemMessage(content="你是一个n8n工作流调试专家，能够根据验证错误提供修正方案。"),
            HumanMessage(content=feedback_prompt)
        ])

        return self._extract_json_from_response(response.content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def deploy_to_n8n(self, workflow_json: str) -> Dict:
        """部署工作流到n8n"""
        url = f"{self.n8n_base_url}/api/v1/workflows"
        headers = {
            "Content-Type": "application/json",
            "X-N8N-API-KEY": self.n8n_api_key
        }

        try:
            # 先进行最终验证
            validation_result = self.validate_schema(workflow_json)
            if not validation_result:
                return {
                    "success": False,
                    "error": f"部署前验证失败: {', '.join(validation_result.errors)}",
                    "stage": "pre_deployment_validation"
                }

            workflow_data = json5.loads(workflow_json)
            # 异步发送HTTP请求（替换requests.post为aiohttp.post）
            response = await self.session.post(
                url,
                json=workflow_data,
                headers=headers,
                timeout=30  # 异步超时设置
            )
            # 异步解析响应JSON
            response_data = await response.json()
            print(f"workflow_data: {workflow_data}")
            print("rrrrrrrrr:", response_data)

            if response.status in [200, 201]:
                # 获取创建的工作流ID进行二次验证
                workflow_id = response_data.get('id')
                if workflow_id:
                    # 验证工作流是否真正可用
                    verify_result = await self.verify_workflow(workflow_id)
                    print("verify workflow:", verify_result)
                    if not verify_result['success']:
                        # 删除有问题的工作流
                        await  self.delete_workflow(workflow_id)
                        return {
                            "success": False,
                            "error": f"工作流创建成功但验证失败: {verify_result['error']}",
                            "workflow_id": workflow_id
                        }
                return {
                    "success": True,
                    "data": response_data,
                    "stage": "deployment"
                }
            else:
                error_msg = f"部署失败: {response.status} - {response_data.get('message')}"
                # 学习新的错误模式
                self._learn_from_error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "stage": "deployment"
                }

        except aiohttp.ClientError as e:
            error_msg = f"网络请求错误: {str(e)}"
            return {"success": False, "error": error_msg, "stage": "network"}
        except json.JSONDecodeError as e:
            error_msg = f"JSON解析错误: {str(e)}"
            return {"success": False, "error": error_msg, "stage": "parsing"}

    async def verify_workflow(self, workflow_id: str) -> Dict:
        """验证工作流是否真正可用"""
        url = f"{self.n8n_base_url}/api/v1/workflows/{workflow_id}/activate"
        headers = {"X-N8N-API-KEY": self.n8n_api_key, "Content-Type": "application/json"}

        try:
            response = await self.session.post(url, json={}, headers=headers, timeout=10)
            response_data = await response.json()
            print(f"verify response: {response_data}")
            if response.status == 200:
                # 可额外检查响应中的active字段是否为true，确认激活成功
                if response_data.get("active") is True:
                    return {"success": True, "message": "工作流激活成功"}
                else:
                    return {"success": False, "error": f"工作流状态未变为激活:{response_data.get("message")}"}
            return {"success": False, "error": f"获取工作流失败:{response.status} - {response_data.get("message")}"}
        except Exception as e:
            return {"success": False, "error": f"验证错误: {str(e)}"}

    async def delete_workflow(self, workflow_id: str):
        """删除工作流"""
        url = f"{self.n8n_base_url}/api/v1/workflows/{workflow_id}"
        headers = {"X-N8N-API-KEY": self.n8n_api_key}

        try:
            await self.session.delete(url, headers=headers, timeout=10)
        except aiohttp.ClientError:
            pass

    def _learn_from_error(self, error_message: str):
        """从错误中学习新的模式"""
        # 检查是否已经是已知模式
        existing_pattern = self.match_error_pattern(error_message)
        if not existing_pattern:
            # 提取错误关键词
            keywords = re.findall(r'\b[a-zA-Z]{4,}\b', error_message)
            if keywords:
                new_pattern = ErrorPattern(
                    error_pattern=f".*{'|'.join(keywords[:3])}.*",
                    solution=f"处理{', '.join(keywords[:2])}相关错误",
                    priority=2
                )
                self.error_patterns.append(new_pattern)
                logger.info(f"学习到新的错误模式: {new_pattern.error_pattern}")

    def _clean_workflow_json(self, workflow_config: Dict) -> Dict:
        """清理和规范化工作流 JSON，确保其符合 n8n 要求"""
        print("\n步骤4: 清理和规范化工作流 JSON...")

        # 验证节点和连接
        if not workflow_config.get("nodes"):
            raise ValueError("工作流必须包含至少一个节点")

        if not workflow_config.get("connections"):
            raise ValueError("工作流必须包含节点之间的连接关系")

        print("清理前的工作流信息:")
        print(f"  节点数量: {len(workflow_config.get('nodes', []))}")
        print(f"  连接数量: {len(workflow_config.get('connections', {}))}")

        nodes = workflow_config.get("nodes", [])

        for i, node in enumerate(nodes):
            print(f"  清理节点 {i + 1}: {node.get('name', 'Unknown')} ({node.get('type', 'Unknown')})")
            parameters = node.get("parameters", {})

            # 验证参数是否符合规范
            node_type = node.get("type")
            # current_version = node.get("typeVersion")

            if node_type in self.base_nodes:
                node_spec = self.base_nodes[node_type]
                for param_name, param_value in parameters.items():
                    # 查找对应的参数规范
                    param_spec = None
                    for spec_param in node_spec.get('parameters', []):
                        if spec_param['name'] == param_name:
                            param_spec = spec_param
                            break

                    if param_spec:
                        # 检查options类型参数的值是否在允许范围内
                        if param_spec.get('type') == 'options' and param_spec.get('options'):
                            if param_value not in param_spec['options']:
                                print(f"    参数 {param_name} 的值 '{param_value}' 不在允许范围内")
                                print(f"      允许的值: {param_spec['options']}")
                                # 使用默认值或第一个允许值
                                if param_spec.get('default') in param_spec['options']:
                                    parameters[param_name] = param_spec['default']
                                    print(f"      已修正为默认值: {param_spec['default']}")
                                else:
                                    parameters[param_name] = param_spec['options'][0]
                                    print(f"      已修正为第一个允许值: {param_spec['options'][0]}")

            # 1. 将空的 'options: {}' 转换为 'options: []'
            if "options" in parameters and isinstance(parameters["options"], dict) and not parameters["options"]:
                parameters["options"] = []  # 转换为空列表
                print("    转换空options对象为数组")

            # 2. 确保集合类型参数是 [] 而不是 {}
            for param_name, param_value in parameters.items():
                if isinstance(param_value, dict) and not param_value:
                    # 检查是否是 n8n 中常见的集合类型参数
                    if param_name in ["bodyParameters", "conditions"]:
                        # 检查其子字段
                        if "values" in param_value and isinstance(param_value["values"], dict) and not param_value[
                            "values"]:
                            param_value["values"] = []
                            print(f"    转换 {param_name}.values 为空数组")
                        if "boolean" in param_value and isinstance(param_value["boolean"], dict) and not param_value[
                            "boolean"]:
                            param_value["boolean"] = []
                            print(f"    转换 {param_name}.boolean 为空数组")

            node["parameters"] = parameters
        workflow_config["nodes"] = nodes

        # 移除 n8n API 创建工作流时不允许的顶层属性
        properties_to_remove = ["versionId", "id", "staticData", "meta", "pinData", "createdAt", "updatedAt",
                                "triggerCount","tags"]
        for prop in properties_to_remove:
            if prop in workflow_config:
                del workflow_config[prop]
                print(f"  移除不允许的属性: {prop}")

        # 移除 'active' 属性，因为它在创建时通常是只读的或不允许设置
        if "active" in workflow_config:
            del workflow_config["active"]
            print("  移除active属性")

        print("工作流 JSON 清理完成")
        return workflow_config


def create_workflow_agent():
    """创建LangGraph智能体"""
    from langgraph.graph import StateGraph, END

    agent = N8NWorkflowAgent()

    # 创建图
    workflow = StateGraph(AgentState)

    # 定义节点 - 生成工作流
    async def generate_node(state: AgentState):
        """生成工作流节点"""
        current_iteration = state.iteration_count
        logger.info(f"第{state.iteration_count + 1}次生成工作流...")

        workflow_json = await agent.generate_workflow(
            state.user_input,
            state.error_messages
        )

        # 进行多阶段验证
        validation_results = []

        # 语法验证
        syntax_result = agent.validate_syntax(workflow_json)
        validation_results.append(syntax_result)

        if syntax_result:
            # 模式验证
            schema_result = agent.validate_schema(workflow_json)
            validation_results.append(schema_result)

            if schema_result:
                # 语义验证
                semantic_result = agent.validate_semantics(workflow_json, state.user_input)
                validation_results.append(semantic_result)

        return {
            "workflow_json": workflow_json,
            "iteration_count": current_iteration + 1,
            "validation_results": validation_results,
            "current_stage": "generation"
        }

    # 定义节点 - 部署工作流（依赖agent实例的_clean_workflow_json方法）
    async def deploy_node(state: AgentState):
        """部署工作流节点"""
        logger.info("部署工作流到n8n...")
        current_iteration = state.iteration_count
        # 解析生成的工作流JSON
        workflow1 = json5.loads(state.workflow_json)
        # 调用agent的清理方法（修正：通过agent实例调用，而非全局函数）
        stared_workflow = agent._clean_workflow_json(workflow1)

        # 保存工作流
        print("\n步骤5: 保存工作流配置到config目录...")
        try:
            save_dir = os.path.join(os.path.dirname(__file__), 'config')
            os.makedirs(save_dir, exist_ok=True)
            print(f"  确保目录存在: {save_dir}")

            # 生成文件名（使用时间戳避免重复）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_file_path = os.path.join(save_dir, f"{timestamp}.json")

            # 只保存n8n工作流配置（修正：正确调用json.dump）
            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(stared_workflow, f, ensure_ascii=False, indent=2)

            print(f"  工作流配置已保存到: {config_file_path}")

        except Exception as e:
            print(f"  保存工作流配置失败: {e}")

        # 部署工作流（修正：将清理后的字典转为JSON字符串）
        result = await agent.deploy_to_n8n(json.dumps(stared_workflow, ensure_ascii=False, indent=2))
        print(f"result: {result}")
        if result['success']:
            logger.info("部署成功!")
            return {
                "deployment_result": result,
                "iteration_count": current_iteration,
                "current_stage": "deployment_success"
            }
        else:
            logger.warning(f"部署失败: {result['error']}")
            return {
                "deployment_result": result,
                "error_messages": state.error_messages + [result['error']],
                "iteration_count": state.iteration_count ,
                "current_stage": "deployment_failed"
            }

    def should_continue(state: AgentState):
        """决定是否继续迭代"""
        if state.deployment_result.get('success', False):
            logger.info("工作流部署成功，结束流程")
            return END
        elif state.iteration_count >= state.max_iterations:
            logger.warning("达到最大迭代次数，结束流程")
            return END
        else:
            # 检查验证结果，如果有严重错误直接重新生成
            for validation_result in state.validation_results:
                if validation_result.errors:
                    logger.info("验证发现错误，重新生成...")
                    return "generate"

            logger.info("继续部署流程...")
            return "generate"

    # 添加节点和边
    workflow.add_node("generate", generate_node)
    workflow.add_node("deploy", deploy_node)

    workflow.set_entry_point("generate")
    workflow.add_conditional_edges(
        "generate",
        # 修正：条件判断逻辑（确保workflow_json存在）
        lambda state: "deploy" if state.workflow_json.strip() else "generate",
        {
            "generate": "generate",
            "deploy": "deploy"
        }
    )

    workflow.add_conditional_edges(
        "deploy",
        should_continue,
        {
            "generate": "generate",
            END: END
        }
    )

    return workflow.compile(), agent


# 使用示例
async def main():
    # 创建智能体
    app,agent = create_workflow_agent()

    # 初始状态
    initial_state = AgentState(
        user_input = "创建一个每日销售报告工作流，从CRM、支付系统和分析平台获取数据，然后发送邮件报告",
        #user_input="创建一个 n8n 工作流，当有新订单进入时触发，更新我们数据库中的库存，向客户发送确认邮件，并生成运输标签。",
            workflow_json = "",
        deployment_result = {},
        error_messages = [],
        iteration_count = 0,
        max_iterations = 5,
        validation_results = [],
        current_stage = "start"
    )

    # 执行智能体
    try:
        final_state = await app.ainvoke(initial_state)
        # 🔵 关键：校验 final_state 类型，确保是 AgentState 对象
        if not isinstance(final_state, AgentState):
            # 若为字典，尝试转为 AgentState（兼容意外转换的情况）
            if isinstance(final_state, dict):
                final_state = AgentState(**final_state)
            else:
                raise TypeError(f"final_state 类型错误，应为 AgentState，实际为 {type(final_state)}")

        if final_state.deployment_result.get('success', False):
            print("✅ 工作流部署成功!")
            # 删除成功部署的工作流 然后合理的json给ts再进行统一部署处理
            workflow_id = final_state.deployment_result['data'].get('id')
            workflow_json = final_state.workflow_json
            await  agent.delete_workflow(workflow_id)
            print("======>>>>>",workflow_json)
            print(final_state.get("workflow_json"))
            print(f"工作流ID: {final_state.deployment_result['data'].get('id', '未知')}")
            print(f"总迭代次数: {final_state.iteration_count}")
        else:
            print("❌ 工作流部署失败")
            print(f"最终错误: {final_state.deployment_result.get('error', '未知错误')}")
            print(f"总共尝试次数: {final_state.iteration_count}")
            print(f"错误历史: {final_state.error_messages}")

    except Exception as e:
        print(f"智能体执行出错: {e}")
    finally:
        # 🔵 无论成功失败，都关闭会话和连接池
        if agent:
            # 关闭会话（会释放所有连接）
            if not agent.session.closed:
                await agent.session.close()
            # 关闭连接池
            await agent.connector.close()
            print("已关闭所有异步连接和会话")


if __name__ == "__main__":
    asyncio.run(main())
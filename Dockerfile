# 使用Ubuntu 22.04作为基础镜像
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /app

# 设置非交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 安装Python、pip和Node.js
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 验证Node.js安装
RUN node --version && npm --version

# 设置conda环境路径
ENV PATH="/opt/conda/envs/n8n_env/bin:$PATH"
ENV CONDA_DEFAULT_ENV=n8n_env
ENV CONDA_PREFIX="/opt/conda/envs/n8n_env"

# 复制整个conda环境
COPY mcp-n8n-workflow-builder/conda_env/ /opt/conda/envs/n8n_env/

# 一次性复制所有项目文件
COPY . /app/

# 安装依赖并构建项目
WORKDIR /app/mcp-n8n-workflow-builder
RUN npm install && npm run build
WORKDIR /app

# 复制启动脚本
COPY start-services.sh /app/start-services.sh
RUN chmod +x /app/start-services.sh

# 设置环境变量
ENV BACKEND_PORT=8003
ENV PYTHONPATH=/app/WebwithMCP-main/backend
ENV NODE_ENV=production

# 暴露端口
EXPOSE 8003

# 设置健康检查（使用Python替代curl）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/api/health')" || exit 1

# 启动命令
CMD ["/app/start-services.sh"]

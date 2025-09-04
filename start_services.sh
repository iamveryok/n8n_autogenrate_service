#!/bin/bash

# 激活conda环境
export PATH="/opt/conda/envs/n8n_env/bin:$PATH"
export CONDA_DEFAULT_ENV=n8n_env
export CONDA_PREFIX="/opt/conda/envs/n8n_env"

# 设置环境变量
export BACKEND_PORT=8003
export PYTHONPATH=/app/WebwithMCP-main/backend
export NODE_ENV=production

echo "启动服务..."

# 启动MCP服务（后台运行）
cd /app/mcp-n8n-workflow-builder
node build/index.js &
MCP_PID=$!

# 启动FastAPI服务（后台运行）
cd /app/WebwithMCP-main/backend
python main.py &
API_PID=$!

echo "服务已启动:"
echo "  - MCP服务: PID $MCP_PID (端口 3456)"
echo "  - FastAPI服务: PID $API_PID (端口 8003)"

# 等待服务运行
wait $MCP_PID $API_PID

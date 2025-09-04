OPENROUTER_CONFIG = {
    "api_key": "sk-d55e63ecb0534036972a3fe3e1370687",  # 请替换为你的 OpenRouter/DeepSeek API Key
    "api_base": "https://api.deepseek.com/v1",
    "model": "deepseek-chat"
}

N8N_CONFIG = {
    "base_url": "http://localhost:5678",
    "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJmM2RlOTUxZS1jYWQ0LTRiZGUtOTE2Ni1jMWUwNGM5NDY4MDYiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwiaWF0IjoxNzU0ODc4MzEyLCJleHAiOjE3NTczOTA0MDB9._iYsrhu2XAUs_SfknBdk37hT2o_YZ3v1bg1sV7B-GpY"
}

# 获取n8n API密钥的步骤：
# 1. 登录n8n网页界面 (http://localhost:5678)
# 2. 点击右上角的用户图标
# 3. 选择"Settings"（设置）
# 4. 在左侧菜单中选择"API"
# 5. 点击"Create New API Key"（创建新的API密钥）
# 6. 给API密钥起个名字（比如"workflow-generator"）
# 7. 复制生成的API密钥
# 8. 将API密钥粘贴到上面的N8N_CONFIG["api_key"]中
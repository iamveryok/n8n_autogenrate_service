# 指定基础映像
FROM node:22-alpine

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY package*.json ./
RUN npm install

# 复制应用代码到容器
COPY . .

# 暴露应用端口（如果需要）
EXPOSE 3456

# 容器启动命令
CMD ["npm", "start"]
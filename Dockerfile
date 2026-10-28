FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 先复制依赖清单，利用 Docker 缓存加速构建
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制项目所有代码
COPY . .

# 暴露端口（和 main_api.py 里的 8000 对应）
EXPOSE 8000

# 启动命令
CMD ["python", "main_api.py"]
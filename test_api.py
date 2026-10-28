import requests

url = "http://localhost:8000/chat"
payload = {"question": "LoRA的秩r通常设置为多少？"}

print("正在发送请求...")
try:
    response = requests.post(url, json=payload)
    # 打印服务器返回的 JSON 结果
    print("服务器返回：")
    print(response.json())
except Exception as e:
    print("出错了：", e)
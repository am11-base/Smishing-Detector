import requests

url = 'https://smishguard.onrender.com/predict_api'
r = requests.post(url,json={"sms":"hello there"})

print(r.json())
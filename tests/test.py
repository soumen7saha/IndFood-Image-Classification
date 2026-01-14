import requests
import json

url = 'http://localhost:9696/predict_food'
request = {
    "img_url": "data/test/bhindi_masala.jpg",
    "model": "resnet"
}

response = requests.post(url, json=request)

print('_______________________________________')
print(f"Status Code: {response.status_code}")
print(f"Response Text: '{response.text}'")
print('_______________________________________\n')

result = response.json()
t1_c = result['t1_class']
t5_c = list(result['t5_preds'].keys())

print(f"Top predicted class: {t1_c}")
print()
print(f"Top 5 predictions:")
for cls in t5_c:
    print(f"{cls}")
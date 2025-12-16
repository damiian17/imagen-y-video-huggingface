import requests

url = "http://127.0.0.1:8000/edit-smile"
files = {'image': open('input1.png', 'rb')}
data = {'style': 'hollywood'}

print("Sending request to backend...")
try:
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        with open('verification_result.png', 'wb') as f:
            f.write(response.content)
        print("Success! Image saved to verification_result.png")
    else:
        print(f"Failed: {response.status_code} - {response.text}")
except Exception as e:
    print(f"Error: {e}")

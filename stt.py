import json
import requests

data = open("C:/Users/shfkf/Desktop/test.wav", "rb")  # STT를 진행하고자 하는 음성 파일

Lang = "Kor"  # Kor / Jpn / Chn / Eng
URL = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + Lang

ID = "dur49y6olu"  # 인증 정보의 Client ID
Secret = "vXR71moPrN6Q5DHDJ7FZRNQocfId8tBpJCknMW2h"  # 인증 정보의 Client Secret

headers = {
    "Content-Type": "application/octet-stream",  # Fix
    "X-NCP-APIGW-API-KEY-ID": ID,
    "X-NCP-APIGW-API-KEY": Secret,
}
response = requests.post(URL, data=data, headers=headers)
rescode = response.status_code

if (rescode == 200):
    print(response.text)
else:
    print("Error : " + response.text)
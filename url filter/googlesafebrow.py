import requests
def checksafe(url):
    key="AIzaSyA594VV48pUlNR_HgQqAHVrysw-rZ0aqUo"
    URL="https://safebrowsing.googleapis.com/v4/threatMatches:find?key="+key
    data={
        "client": {
        "clientId":"smishguard",
        "clientVersion": "1.0.0"
        },
        "threatInfo": {
        "threatTypes":      ["MALWARE", "SOCIAL_ENGINEERING"],
        "platformTypes":    ["ANY_PLATFORM"],
        "threatEntryTypes": ["URL"],
        "threatEntries": [
            {"url":url}
        ]
        }
    }
    try:
     postreq=requests.post(url=URL,json=data)
    except requests.exceptions.HTTPError as e:
        print (e.response.text)
    else:
     response=postreq.json()
     print(response)
     if(len(response)==0):
        print("Further check needed")
     else: print("Unsafe")

url=input("Enter the url")
checksafe(url)
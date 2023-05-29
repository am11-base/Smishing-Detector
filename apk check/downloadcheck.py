import requests
url="https://drive.google.com/uc?id=1QOmVDpd8hcVYqqUXDXf68UMDWQZP0wQV&export=download"
url1="https://www.apkonline.net/myapkdownloader/apk-downloads/com.kiloo.subwaysurf-01.apk"
response = requests.get(url1, stream=True)
content_type = response.headers.get("content-type")
print(content_type)
# Check if the content type indicates a downloadable file
if "application" in content_type or "pdf" in content_type or "zip" in content_type:
    print("The URL is downloading a file.")
else:
    print("The URL is not downloading a file.")

# Close the response
response.close()

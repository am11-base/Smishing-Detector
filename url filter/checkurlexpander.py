import re
import urlexpander
def http_check(line):
    http = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line)
    if not http:
        return 0
    else:
        return(http)
        
def expander(url):
    return(urlexpander.expand(url))
        

userinput=input("Enter the SMS: ")
url=http_check(userinput)
print("Shortened: "+url[0])
if url[0] !=0:
    expanded=expander(url[0])
    print("Expanded : "+expanded)

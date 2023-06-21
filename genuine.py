import re
emailPattern = re.compile(("([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`" "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|" "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"))
def email_check(line):
    if emailPattern.search(line) is not None:
        return 1
    else:
        return 0
    
def http_check(line):
    http =re.findall(r'(https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|(?:www\.|(?:[a-zA-Z]{1}[-\w]+\.[a-zA-Z]{2,}))(?:[-\w./?%&=]*)?)', line)
    print(http)
    if not http:
        return 0
    else:
        return 1
    
phonePattern = re.compile(r'(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$')

def phoneNumber_check(line):
    if phonePattern.search(line) is not None:
        return 1
    else:
        return 0

user_input=input("Enter the SMS: ")
if(email_check(user_input) or http_check(user_input) or phoneNumber_check(user_input)) :
    print("Further check needed")
else :
    print("Genuine") 
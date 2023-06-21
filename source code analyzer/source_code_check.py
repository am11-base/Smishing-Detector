import os
import time
from selenium import webdriver
from urllib.parse import urlparse
from selenium.webdriver.common.by import By
os.environ['PATH']+=r"C:/SeleniumDriver"

driver = webdriver.Chrome()
def get_subpage_source(url):
    #return 1 if smishing
    driver.get(url)
    source = driver.page_source
    if "<input" in source and ("type=\"text\"" in source or "type=\"email\"" in source or "type=\"password\"" in source):
         url_domain = urlparse(url).hostname
         html_domain = urlparse(driver.current_url).hostname
         print(url_domain,html_domain)
         if url_domain==html_domain:
             return 0
         else:
             return 1

    else:
        return 0
    
start_url = 'https://tkmce.etlab.in/user/login'
present=get_subpage_source(start_url)
print(present)
driver.quit()
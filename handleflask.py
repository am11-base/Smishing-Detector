import numpy as np
import requests 
import re
import warnings
import nltk
import pickle
#nltk.download('stopwords')
from nltk.corpus import stopwords
#
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import scipy.sparse
import urlexpander
import editdistance
import whois
import tldextract
from urllib.parse import urlparse
from datetime import datetime
import os
import time
from selenium import webdriver
from urllib.parse import urlparse
from selenium.webdriver.common.by import By
import urllib3
from selenium.webdriver.chrome.options import Options
from requests.exceptions import RequestException
from flask import Flask, request, jsonify

app=Flask(__name__)
#genuine check

def genuinecheck(sms):
    if(email_check(sms) or http_check(sms) or phoneNumber_check(sms)) :
        return 1 #returns 1 if sms contain any email,link,no
    else:
        return 0

def email_check(line):
    emailPattern = re.compile(("([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`" "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|" "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"))
    if emailPattern.search(line) is not None:
        return 1
    else:
        return 0
    
def http_check(line):
    http =re.findall(r'(https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|(?:www\.|(?:[a-zA-Z]{1}[-\w]+\.[a-zA-Z]{2,}))(?:[-\w./?%&=]*)?)', line)
    if not http:
        return 0
    else:
        return 1
    
def phoneNumber_check(line):
    phonePattern = re.compile(r'(\d{3})\D*(\d{3})\D*(\d{4})\D*(\d*)$')
    if phonePattern.search(line) is not None:
        return 1
    else:
        return 0


#content analyzer module
warnings.filterwarnings('ignore', category=DeprecationWarning)
def clean_text(text):
    stopwords = nltk.corpus.stopwords.words('english')
    ps = nltk.WordNetLemmatizer()
    tokens = re.split('\W+', text)
    stems = [ps.lemmatize(word) for word in tokens if word not in stopwords] # Remove Stopwords
    return stems

def content_analyze(sms):
    
   
    with open('vectorizer.pkl', 'rb') as file:
        loaded_vectorizer = pickle.load(file)
   
    vector = scipy.sparse.csr_matrix(loaded_vectorizer.transform([sms]))
   
    filename = 'contentmodel.pkl'
    with open(filename, 'rb') as file:
        model=pickle.load(file)
        return(model.predict(vector))
    #print(vector)
    


################################################urlfilter module#######################
def url_extract(sms):
    http =re.findall(r'(https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|(?:www\.|(?:[a-zA-Z]{1}[-\w]+\.[a-zA-Z]{2,}))(?:[-\w./?%&=]*)?)',sms)
    #print(http)
    if len(http)==0:
        return None
    else:
        return http[0]




def url_filter(url):
    #return 0 if smishing URL
    #url=preprocess_url(url)
    #print(url)
    expanded_url=expander(url)
    #print(expanded_url)
    if(checksafe(expanded_url)==0):
        print("not safe as per api")
        return 0
    else:
        ml_filter=url_Ml(expanded_url)
        print("response from ml",ml_filter)
        if(ml_filter==[1]):
            print("not safe as per model")
            return 0
        else:
            return 1
        

def preprocess_url(url):
    # Check if the URL starts with a scheme (e.g., http:// or https://)
    if not re.match(r'^\w+://', url):
        # Add 'https://' as the default scheme
        url = 'https://' + url

    return url

def expander(url):
    #expand the URL
    return(urlexpander.expand(url))

def checksafe(url):
    #returns 0 if not safe as per googlesafebrowsing API
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
        print ("Error"+e.response.text)
    else:
     response=postreq.json()
     #print(response)
     if(len(response)==0):
        return 1
     else: return 0


#token extraction
def token_extraction(text):
    text = text.strip()
    tokens = re.split('\W+|_', text)
    count=0
    if text == "":
        return 0,[]
    else:
        for token in tokens:
            if token:
                count= count+1
            else:
                tokens.remove('')
    if count == 0:
        return 0,[]
    return count,tokens

#TLD Count
def tld_count(tokens):
    count=0
    tld_list=['.com', '.net', '.org', '.io', '.co', '.ai', '.co.uk', '.ca', '.dev', '.me', '.de', '.app', '.in', '.is', '.eu', '.gg', '.to', 
'.ph', '.nl', '.id', '.inc', '.website', '.xyz', '.club', '.online', '.info', '.store', '.best', '.live', '.us', '.tech', '.pw', '.pro', '.uk', '.tv', '.cx', '.mx', '.fm', '.cc', '.world', '.space', '.vip', '.life', '.shop', '.host', '.fun', '.biz', '.icu', '.design', '.art'] 
    for i in tokens:
        if i in tld_list:
            count=count+1
    return count

#ratio of digits
def ratio_digits(url):
    if len(url) == 0:
        return 0
    else:
        return len(re.sub("[^0-9]", "", url))/len(url)

#punycodecheck   
def punycode(url):
    if url.startswith("http://xn--") or url.startswith("https://xn--") or 'xn--' in url:
        return 1
    else:
        return 0

#port check
def port(url):
    if re.search("^[a-z][a-z0-9+\-.]*://([a-z0-9\-._~%!$&'()*+,;=]+@)?([a-z0-9\-._~%]+|\[[a-z0-9\-._~%!$&'()*+,;=:]+\]):([0-9]+)",url):
        return 1
    return 0


#ip_present_url
def contains_ip_address(url):
    ip_regex = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    match=re.search(ip_regex, url)
    if match:
        return 1
    else:
        return 0                
#www count
def www_count(url):
    return url.count("www")

#https_count
def https_count(url):
    return url.count("https")

#.com count
def com_count(url):
    return url.count(".com")



#suspecious tld check

def suspecious_tld(tld):
   suspecious_tlds = ['fit','tk', 'gp', 'ga', 'work', 'ml', 'date', 'wang', 'men', 'icu', 'online', 'click', # Spamhaus
        'country', 'stream', 'download', 'xin', 'racing', 'jetzt',
        'ren', 'mom', 'party', 'review', 'trade', 'accountants',
        'science', 'work', 'ninja', 'xyz', 'faith', 'zip', 'cricket', 'win',
        'accountant', 'realtor', 'top', 'christmas', 'gdn', # Shady Top-Level Domains
        'link','zw','bd','ke','pw','quest','support','rest','casa', # Blue Coat Systems
        'asia', 'club', 'la', 'ae', 'exposed', 'pe', 'go.id', 'rs', 'k12.pa.us', 'or.kr',
        'ce.ke', 'audio', 'gob.pe', 'gov.az', 'website', 'bj', 'mx', 'media', 'sa.gov.au' # statistics
        ]

   if tld in suspecious_tlds:
       return 1
   return 0

#brand name check
def brandname_check(tokens):
    count=0
    brand_list=['accenture', 'activisionblizzard', 'adidas', 'adobe', 'adultfriendfinder', 'agriculturalbankofchina', 'akamai', 'alibaba', 'aliexpress', 'alipay', 'alliance', 'alliancedata', 'allianceone', 'allianz', 'alphabet', 'amazon', 'americanairlines', 'americanexpress', 'americantower', 'andersons', 'apache', 'apple', 'arrow', 'ashleymadison', 'audi', 'autodesk', 'avaya', 'avisbudget', 'avon', 'axa', 'badoo', 'baidu', 'bankofamerica', 'bankofchina', 'bankofnewyorkmellon', 'barclays', 'barnes', 'bbc', 'bbt', 'bbva', 'bebo', 'benchmark', 'bestbuy', 'bing', 'biogen', 'blackstone', 'blogger', 'blogspot', 'bmw', 'bnpparibas', 'boeing', 'booking', 'broadcom', 'burberry', 'caesars', 'canon', 'cardinalhealth', 'carmax', 'carters', 'caterpillar', 'cheesecakefactory', 
'chinaconstructionbank', 'cinemark', 'cintas', 'cisco', 'citi', 'citigroup', 'cnet', 'coca-cola', 'colgate', 'colgate-palmolive', 'columbiasportswear', 'commonwealth', 'communityhealth', 'continental', 'dell', 'deltaairlines', 'deutschebank', 'disney', 'dolby', 'dominos', 'donaldson', 'dreamworks', 'dropbox', 'eastman', 'eastmankodak', 'ebay', 'edison', 'electronicarts', 'equifax', 'equinix', 'expedia', 'express', 'facebook', 'fedex', 'flickr', 'footlocker', 'ford', 'fordmotor', 'fossil', 'fosterwheeler', 'foxconn', 'fujitsu', 'gap', 'gartner', 'genesis', 'genuine', 'genworth', 'gigamedia', 'gillette', 'github', 'global', 'globalpayments', 'goodyeartire', 'google', 'gucci', 'harley-davidson', 'harris', 'hewlettpackard', 'hilton', 'hiltonworldwide', 'hmstatil', 'honda', 'hsbc', 'huawei', 'huntingtonbancshares', 'hyundai', 'ibm', 'ikea', 'imdb', 'imgur', 'ingbank', 'insight', 'instagram', 'intel', 'jackdaniels', 'jnj', 'jpmorgan', 'jpmorganchase', 'kelly', 'kfc', 'kindermorgan', 'lbrands', 'lego', 'lennox', 'lenovo', 'lindsay', 'linkedin', 'livejasmin', 'loreal', 'louisvuitton', 'mastercard', 'mcdonalds', 'mckesson', 'mckinsey', 'mercedes-benz', 'microsoft', 'microsoftonline', 'mini', 'mitsubishi', 'morganstanley', 'motorola', 'mrcglobal', 'mtv', 'myspace', 'nescafe', 'nestle', 'netflix', 'nike', 'nintendo', 'nissan', 'nissanmotor', 'nvidia', 'nytimes', 'oracle', 'panasonic', 'paypal', 'pepsi', 'pepsico', 'philips', 'pinterest', 'pocket', 'pornhub', 'porsche', 'prada', 'rabobank', 'reddit', 'regal', 'royalbankofcanada', 'samsung', 'scotiabank', 'shell', 'siemens', 'skype', 'snapchat', 'sony', 'soundcloud', 'spiritairlines', 'spotify', 'sprite', 'stackexchange', 'stackoverflow', 'starbucks', 'swatch', 'swift', 'symantec', 'synaptics', 'target', 'telegram', 'tesla', 'teslamotors', 'theguardian', 'homedepot', 'piratebay', 'tiffany', 'tinder', 'tmall', 'toyota', 'tripadvisor', 'tumblr', 'twitch', 'twitter', 'underarmour', 'unilever', 'universal', 'ups', 'verizon', 'viber', 'visa', 'volkswagen', 'volvocars', 'walmart', 
'wechat', 'weibo', 'whatsapp', 'wikipedia', 'wordpress', 'yahoo', 'yamaha', 'yandex', 'youtube', 'zara', 'zebra', 'iphone', 'icloud', 'itunes', 'sinara', 'normshield', 'bga', 'sinaralabs', 'roksit', 'cybrml', 'turkcell', 'n11', 'hepsiburada', 'migros']  
    for i in tokens:
        if i in brand_list:
            count=count+1
    return count

#check brand_typosquatting in url
def check_typosquatting_brand(tokens):
    brand_list=['accenture', 'activisionblizzard', 'adidas', 'adobe', 'adultfriendfinder', 'agriculturalbankofchina', 'akamai', 'alibaba', 'aliexpress', 'alipay', 'alliance', 'alliancedata', 'allianceone', 'allianz', 'alphabet', 'amazon', 'americanairlines', 'americanexpress', 'americantower', 'andersons', 'apache', 'apple', 'arrow', 'ashleymadison', 'audi', 'autodesk', 'avaya', 'avisbudget', 'avon', 'axa', 'badoo', 'baidu', 'bankofamerica', 'bankofchina', 'bankofnewyorkmellon', 'barclays', 'barnes', 'bbc', 'bbt', 'bbva', 'bebo', 'benchmark', 'bestbuy', 'bing', 'biogen', 'blackstone', 'blogger', 'blogspot', 'bmw', 'bnpparibas', 'boeing', 'booking', 'broadcom', 'burberry', 'caesars', 'canon', 'cardinalhealth', 'carmax', 'carters', 'caterpillar', 'cheesecakefactory', 
'chinaconstructionbank', 'cinemark', 'cintas', 'cisco', 'citi', 'citigroup', 'cnet', 'coca-cola', 'colgate', 'colgate-palmolive', 'columbiasportswear', 'commonwealth', 'communityhealth', 'continental', 'dell', 'deltaairlines', 'deutschebank', 'disney', 'dolby', 'dominos', 'donaldson', 'dreamworks', 'dropbox', 'eastman', 'eastmankodak', 'ebay', 'edison', 'electronicarts', 'equifax', 'equinix', 'expedia', 'express', 'facebook', 'fedex', 'flickr', 'footlocker', 'ford', 'fordmotor', 'fossil', 'fosterwheeler', 'foxconn', 'fujitsu', 'gap', 'gartner', 'genesis', 'genuine', 'genworth', 'gigamedia', 'gillette', 'github', 'global', 'globalpayments', 'goodyeartire', 'google', 'gucci', 'harley-davidson', 'harris', 'hewlettpackard', 'hilton', 'hiltonworldwide', 'hmstatil', 'honda', 'hsbc', 'huawei', 'huntingtonbancshares', 'hyundai', 'ibm', 'ikea', 'imdb', 'imgur', 'ingbank', 'insight', 'instagram', 'intel', 'jackdaniels', 'jnj', 'jpmorgan', 'jpmorganchase', 'kelly', 'kfc', 'kindermorgan', 'lbrands', 'lego', 'lennox', 'lenovo', 'lindsay', 'linkedin', 'livejasmin', 'loreal', 'louisvuitton', 'mastercard', 'mcdonalds', 'mckesson', 'mckinsey', 'mercedes-benz', 'microsoft', 'microsoftonline', 'mini', 'mitsubishi', 'morganstanley', 'motorola', 'mrcglobal', 'mtv', 'myspace', 'nescafe', 'nestle', 'netflix', 'nike', 'nintendo', 'nissan', 'nissanmotor', 'nvidia', 'nytimes', 'oracle', 'panasonic', 'paypal', 'pepsi', 'pepsico', 'philips', 'pinterest', 'pocket', 'pornhub', 'porsche', 'prada', 'rabobank', 'reddit', 'regal', 'royalbankofcanada', 'samsung', 'scotiabank', 'shell', 'siemens', 'skype', 'snapchat', 'sony', 'soundcloud', 'spiritairlines', 'spotify', 'sprite', 'stackexchange', 'stackoverflow', 'starbucks', 'swatch', 'swift', 'symantec', 'synaptics', 'target', 'telegram', 'tesla', 'teslamotors', 'theguardian', 'homedepot', 'piratebay', 'tiffany', 'tinder', 'tmall', 'toyota', 'tripadvisor', 'tumblr', 'twitch', 'twitter', 'underarmour', 'unilever', 'universal', 'ups', 'verizon', 'viber', 'visa', 'volkswagen', 'volvocars', 'walmart', 
'wechat', 'weibo', 'whatsapp', 'wikipedia', 'wordpress', 'yahoo', 'yamaha', 'yandex', 'youtube', 'zara', 'zebra', 'iphone', 'icloud', 'itunes', 'sinara', 'normshield', 'bga', 'sinaralabs', 'roksit', 'cybrml', 'turkcell', 'n11', 'hepsiburada', 'migros']  
    for i in tokens:
            if i != 'www' :
                for brand in brand_list:
                    score = editdistance.eval(brand,i)
                    if (score <= 1) and (score != 0) :
                        print(i,brand)
                        return 1
                    else :
                        continue
    return 0

# prefix/suffix check

def prefix_suffix(domain):
    if '-' in domain:
        return 1
    else:
        return 0

# path extension check for .txt,.js,.exe


def path_extension(path_with_query):
    if '.txt' in path_with_query or '.exe' in path_with_query or '.js' in path_with_query:
        return 1
    else:
        return 0

# shortest token length


def shortest_token_length(txt):  # txt = host or paths
    shortest_len = 0
    txt = txt.strip()
    tokens = re.split('\W+|_', txt)
    if txt == "":
        return 0
    else:
        if "" in tokens:
            tokens.remove('')
        shortest_len = len(tokens[0])
        for token in tokens:
            if shortest_len > len(token):
                shortest_len = len(token)
    return shortest_len

# avg token length


def average_token_length(txt):  # txt = url or host or paths
    total = 0
    avg_length = 0
    txt = txt.strip()
    tokens = re.split('\W+|_', txt)
    if txt == "":
        return avg_length
    else:
        if "" in tokens:
            tokens.remove('')
        for token in tokens:
            total += len(token)
        avg_length = total/len(tokens)
    return avg_length

# domain registration age

def find_domain_age(domain):
    try:
        w = whois.whois(domain)

        creation_date = w.creation_date
        if type(creation_date) is list:
            creation_date = creation_date[0]

        today = datetime.now()
        age = (today - creation_date).days

        return age
    except Exception as e:
        return -1
    

#number of digits in url
def count_digits_in_url(url):
    count = 0
    for char in url:
        if char.isdigit():
            count += 1
    return count

#number of dots in url
def count_dots_in_url(url):
    dot_positions = []
    for i in range(len(url)):
        if url[i] == ".":
            dot_positions.append(i)
    return len(dot_positions)

def create_feature_vector(url):
    url_parsed = urlparse(url)
    host = url_parsed.netloc
    path = url_parsed.path
    query = url_parsed.query
    scheme = url_parsed.scheme
    extracted_domain = tldextract.extract(url)
    tld = extracted_domain.suffix
    domain = extracted_domain.domain
    subdomain = extracted_domain.subdomain
    urltok_count,url_tokens=token_extraction(url)

    has_suspecious_tld=suspecious_tld(tld)
    has_ip=contains_ip_address(url)
    url_www_count=www_count(url)
    url_com_count=com_count(url)
    url_https_count=https_count(url)

    url_ratio=ratio_digits(url)
    has_puny=punycode(url)
    has_port=port(url)
    len_url=len(url)
    url_count_dots=count_dots_in_url(url)
    url_count_digits=count_digits_in_url(url)
    #print(url_ratio,has_port,has_puny)
    url_tld_count=tld_count(url_tokens)
    has_brand_typosquatt=check_typosquatting_brand(url_tokens)

    #print(urltok_count,url_tokens)

    brandname_count=brandname_check(url_tokens)

    tmp = url[url.find(extracted_domain.suffix):len(url)]
    pth = tmp.partition("/")
    paths = pth[1] + pth[2]

    has_prefix = prefix_suffix(domain)
    has_path_extension = path_extension(path+'?'+query)

    shortest_token_url = shortest_token_length(url)
    avg_length_token_url = average_token_length(url)


    domain_age = find_domain_age(domain + '.' + tld)
    feature=[
      urltok_count,
            url_tld_count,
            url_ratio,
            has_puny,
            has_port,
            has_ip,
            url_www_count,
            url_https_count,
            url_com_count,
            has_suspecious_tld,
            brandname_count,
            has_brand_typosquatt,
            has_prefix,
            has_path_extension,
            shortest_token_url,
            avg_length_token_url,
            domain_age,
            len_url,
            url_count_digits,
            url_count_dots,
    ]
    feature_vector = np.array(feature)

    return feature_vector

def url_Ml(url):
    feature_vector=create_feature_vector(url)
    reshaped_feature_vector = feature_vector.reshape(1, -1)
    #print("Feature vector",feature_vector)
    filename = 'urlmodel.pkl'
    with open(filename, 'rb') as file:
        model=pickle.load(file)
        return(model.predict(reshaped_feature_vector))


##############################################Source analyser Module####################33333
def source_analyser(url):
    #return 1 if smishing
    os.environ['PATH']+=r"C:/SeleniumDriver"
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Enable headless mode
    chrome_options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=chrome_options)
    from selenium.common.exceptions import WebDriverException
    try:
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
    except WebDriverException as e:
        #print("Exception occurred while processing URL. Treating SMS as legitimate.")
        return 0

###################################APK downloader check
def apk_check(url):
    #return 1 if smishing
    try:
        response = requests.get(url, stream=True)
        content_type = response.headers.get("content-type")
        print(content_type)
        # Check if the content type indicates a downloadable file
        if "application" in content_type or "pdf" in content_type or "zip" in content_type:
            print("The URL is downloading a file.")
            response.close()
            return 1
        else:
            print("The URL is not downloading a file.")
            response.close()
            return 0
    
    except RequestException as e:
        print("A connection error occurred while accessing the URL.")
        return 0

@app.route('/predict_api',methods=['POST'])
def predict_api():
    get_req=request.get_json()
    sms=get_req['sms']
    print("Sms:",sms)
    print(type(sms))
    content_analysis=content_analyze(sms)
    url=url_extract(sms)
    if url is None:
        if genuinecheck(sms)==0:
            response_data = { "message": "Ham"}
            return jsonify(response_data)
        else:
            if(content_analysis==[1]):
                print("Spam message")
                response_data = { "message": "Spam"}
                return jsonify(response_data)
            elif(content_analysis==[0]):
                response_data = { "message": "Ham"}
                return jsonify(response_data)
            else:
                print("Smishing message")
                response_data = { "message": "Smishing"}
                return jsonify(response_data)
            
    else:
        url=preprocess_url(url)
        url_analysis=url_filter(url)
        if(url_analysis==0):
            print("smishing")
            response_data = { "message": "Smishing"}
            return jsonify(response_data)
        else:
            print("checking source")
            
            source_analysis=source_analyser(url)
            if(source_analysis==1):
                print("smishing")
                response_data = { "message": "Smishing"}
                return jsonify(response_data)
            else:
                print("checking apk")
                apk_analysis=apk_check(url)
                if(apk_analysis==1):
                    print("smishing")
                    response_data = { "message": "Smishing"}
                    return jsonify(response_data)
                else:
                    if(content_analysis==[1]):
                        response_data = { "message": "Spam"}
                        return jsonify(response_data)
                    elif(content_analysis==[0]):
                        response_data = { "message": "Ham"}
                        return jsonify(response_data)
                    else:
                        print("Smishing message")
                        response_data = { "message": "Smishing"}
                        return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)
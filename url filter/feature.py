import csv
import re
import editdistance
import whois
import tldextract
from urllib.parse import urlparse
from datetime import datetime

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


urlcsv=open(r"E:\Users\hashim\Documents\cse 19-23\4th yr\project\references\input_url.csv", 'r',encoding="utf8")
features=open(r'E:\Users\hashim\Documents\cse 19-23\4th yr\project\references\code_splits\output.csv', 'a',encoding="utf8",newline='')

csv_url = csv.reader(urlcsv)

for i in range(11257):
    next(csv_url)

writer = csv.writer(features)
#writer.writerow(["url","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","class_label"])
label=0

for urlrow in csv_url:

    url=urlrow[0]
    url_parsed = urlparse(url)
    host = url_parsed.netloc
    path = url_parsed.path
    query = url_parsed.query
    scheme = url_parsed.scheme
    extracted_domain = tldextract.extract(url)
    tld = extracted_domain.suffix
    domain = extracted_domain.domain
    subdomain = extracted_domain.subdomain
    print("URL"+url)
    #print(url_parsed)
    print("Host"+host)
    print("PAth"+path)
    print("Query"+query)
    print("Subdomain"+subdomain)
    print("TLD"+tld)
    print("Domain"+domain)


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
    if(urlrow[1]=="phishing"):
        label=1
    else:
        label=0
    writer.writerow(
        [
            url,
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
            
            label
            ])

urlcsv.close()
features.close()
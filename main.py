from bs4 import BeautifulSoup
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
data = []

def get_page(url):
    try:
        agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
        page = requests.get(url, verify=False, headers=agent)
        return page
    except Exception as e:
        print("Could not get page {}: \n {}".format(url, e))
    return None

def extract_page_attributes(content):
    soup = BeautifulSoup(content, 'html.parser')
    feature_column = soup.find_all("div", attrs={"class":"feature-column"})
    for col in feature_column:
        pageid = col.attrs["data-customer"]
        feature_page = get_page("https://www.rest.co.il/rest/" + pageid)
        feature_soup = soup = BeautifulSoup(feature_page.content, 'html.parser')
        # need to extract features

def get_body_for_pages(num):
    page = get_page("https://www.rest.co.il/restaurants/israel")
    data.append(extract_page_attributes(page.content))
    if(num == 1):
        return 
    for i in range(1,num):
        print("page ", i)
        page = get_page("https://www.rest.co.il/restaurants/israel/page-{}/".format(i))
        if(page is None):
            break
        extract_page_attributes(page.content)


body = get_body_for_pages(2)


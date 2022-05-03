from bs4 import BeautifulSoup
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
data = []

'''
Oren: attributes /num_of_feedbacks
Ilya: name  / stars  /  address

Create a function for each data which will extract the relevant data. 
Those function will be called from 'extract_page_attributes' function

final_box: func(stars + num_feedbacks)....
'''

def get_page_soup(url):
    try:
        agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
        page = requests.get(url, verify=False, headers=agent)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup
    except Exception as e:
        print("Could not get page {}: \n {}".format(url, e))
    return None

def get_page_attributes(page): 
    
    return None

def extract_page_attributes(page):
    feature_column = page.find_all("div", attrs={"class":"feature-column"})
    for col in feature_column:
        pageid = col.attrs["data-customer"]
        print("page id "+pageid)
        feature_page = get_page_soup("https://www.rest.co.il/rest/" + pageid)

        # need to extract features

def get_body_for_pages(num):
    page = get_page_soup("https://www.rest.co.il/restaurants/israel")
    data.append(extract_page_attributes(page))
    if(num == 1):
        return 
    for i in range(1,num):
        print("page ", i)
        page = get_page_soup("https://www.rest.co.il/restaurants/israel/page-{}/".format(i))
        if(page is None):
            break
        extract_page_attributes(page)


body = get_body_for_pages(2)


from bs4 import BeautifulSoup
import requests
import urllib3
from geopy.geocoders import Nominatim
import geopy
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

def get_name(feature_page):
    """

    :param feature_page:
    :return: name of the restaurant
    """
    name = feature_page.find("h1")  # , attrs={"class": "main_banner_content"}
    return name.text.split(',')[0]  # take the name "name, location" and leave only the name

def get_stars(feature_page):
    """

    :param feature_page:
    :return:number of stars
    """
    if(feature_page.find("div", attrs={"class":"reviews_wrap"})):
        stars = feature_page.find("div", attrs={"class":"reviews_wrap"}).find("span")  # , attrs={"class": "main_banner_content"}
    else:
        stars = None
    # print(stars)
    if(stars):
        return stars.text
    else:
        return None

def get_geolocation(feature_page):
    """

    :param feature_page:
    :return: geolocation of rest
    """
    address = feature_page.find("h5")
    geolocator = Nominatim(user_agent="catuserbot")
    location = geolocator.geocode(address.text)
    if(location):
        return location.latitude, location.longitude
    else:
        return None
    # return address.text  # take the address

def extract_page_attributes(page):
    feature_column = page.find_all("div", attrs={"class":"feature-column"})
    for col in feature_column:
        pageid = col.attrs["data-customer"]
        feature_page = get_page_soup("https://www.rest.co.il/rest/" + pageid)
        print("page id " + pageid)
        print("name: " + get_name(feature_page))
        print("stars: {}".format(get_stars(feature_page)))
        print("geolocation: {}".format(get_geolocation(feature_page)))
        print("      ")

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


from attr import attributes
from bs4 import BeautifulSoup
import requests
import urllib3
from geopy.geocoders import Nominatim
import geopy
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
data = []

'''
Oren: attributes /num_of_feedbacks
Ilya: name  / stars  /  address

Create a function for each data which will extract the relevant data. 
Those function will be called from 'extract_page_attributes' function

Empty attributes need to be set to 0

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

def get_page_attributes_sel(url, feature_body):
    driver_path = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(driver_path)
    driver.get(url)

    button = None
    try:
        button = driver.find_element_by_css_selector('#site-main > div > div.place_and_rush_hours > div > div > div > div > small')
    except Exception as e:
        pass
    if(button):
        button.click()
        html = driver.page_source
        soup = BeautifulSoup(html)
        attributes_body = soup.find('div', {'class':'pop-scroll-wrap'}).find_all('li')
        attributes = [attr.text.replace("\n","").strip() for attr in attributes_body]
        driver.quit()
        return attributes
    else:
        driver.quit()
        return get_page_attributes(feature_body)

def get_page_attributes(body):
    attributes_body = body.find('div', {'class':'place_info'}).find_all('li')
    attributes = [attr.text for attr in attributes_body]
    return attributes

def get_number_of_reviews(body):
    try:
        reviews_body = body.find('div', {'class':'raviews_box_item'})
        reviews_link = reviews_body.find('a')
        num = reviews_link.text.split(' ')[0]
        if(num.isdecimal()):
            return num
    except Exception as e:
        print("could not get number of reviews ", e)
    return 0

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
        try:
            pageid = col.attrs["data-customer"]
            print("page id "+pageid)
            url = "https://www.rest.co.il/rest/" + pageid
            feature_body = get_page_soup(url)
            page_attributes = get_page_attributes_sel(url, feature_body)
            print("page_attributes ", page_attributes)
            num_of_reviews = get_number_of_reviews(feature_body)
            print('number of reviews: ', num_of_reviews)
            print("name: " + get_name(feature_page))
            print("stars: {}".format(get_stars(feature_page)))
            print("geolocation: {}".format(get_geolocation(feature_page)))
            # need to extract features
        except Exception as e:
            print("error: ", e)

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
        data.append(extract_page_attributes(page))



body = get_body_for_pages(2)


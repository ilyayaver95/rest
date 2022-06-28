import math
import os
from datetime import datetime
from pathlib import Path
from turtle import pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import urllib3
from bs4 import BeautifulSoup
from geopandas import GeoDataFrame
from geopy import Point
from geopy.geocoders import Nominatim
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from webdriver_manager.chrome import ChromeDriverManager

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
data = []

def get_page_soup(url):
    try:
        agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
        page = requests.get(url, verify=False, headers=agent)
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup
    except Exception as e:
        print("[get_page_soup] Could not get page {}: \n {}".format(url, e))
    return None

def get_page_attributes_sel(url, feature_body):
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    button = None
    try:
        button = driver.find_element_by_css_selector('#site-main > div > div.place_and_rush_hours > div > div > div > div > small')
    except Exception as e:
        pass
    if(button):
        button.click()
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        attributes_body = soup.find('div', {'class':'pop-scroll-wrap'})
        if attributes_body:
            attributes_list = attributes_body.find_all('li')
            attributes = [attr.text.replace("\n","").strip() for attr in attributes_list]
            driver.quit()
            return attributes
        return []
    else:
        driver.quit()
        return get_page_attributes(feature_body)

def get_page_attributes(body):
    attributes_body = body.find('div', {'class':'place_info'})
    if(attributes_body):
        attributes_list = attributes_body.find_all('li')
        attributes = [attr.text for attr in attributes_list]
        return attributes
    return []

def get_type(feature_page):
    """
    gets the type of restaurant: Italian/
    :param body:
    :return:
    """
    try:
        name = feature_page.find("h6")  # , attrs={"class": "main_banner_content"}
        return name.text.split(',')[0].strip()  # take the name "name, location" and leave only the name
    except Exception as e:
        print("[get_type] error: ", e)

def get_number_of_reviews(body):
    try:
        reviews_body = body.find('div', {'class':'raviews_box_item'})
        if(reviews_body):
            reviews_link = reviews_body.find('a')
            if(reviews_link):
                num = reviews_link.text.split(' ')[0]
                if(num.isdecimal()):
                    return num
    except Exception as e:
        print("[get_number_of_reviews] could not get number of reviews ", e)
    return 0

def get_name(feature_page):
    """

    :param feature_page:
    :return: name of the restaurant
    """
    try:
        name = feature_page.find("h1")  # , attrs={"class": "main_banner_content"}
        return name.text.split(',')[0].strip()  # take the name "name, location" and leave only the name
    except Exception as e:
        print("[get_name] error: ", e)

def get_stars(feature_page):
    """

    :param feature_page:
    :return:number of stars
    """
    if(feature_page.find("div", attrs={"class":"reviews_wrap"})):
        stars = feature_page.find("div", attrs={"class":"reviews_wrap"}).find("span")  # , attrs={"class": "main_banner_content"}
    else:
        stars = None
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

def save_df_to_csv(df):
    """
    creates csv based on the DF.
    """
    folder_name = "Resturants Output"
    if(not os.path.exists(folder_name)):
        os.mkdir(folder_name)
    file_name = "Rest df {}.csv".format(datetime.now().strftime("%d.%b.%Y %H-%M-%S"))
    filepath = Path("{}/{}".format(folder_name, file_name))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, encoding = 'utf-8-sig', index=False)
    print("Your df is saved !")

def extract_page_attributes(page):
    """

    :param page:
    :return: list of JSON of restaurants
    """
    feature_column = page.find_all("div", attrs={"class":"feature-column"})
    print("feature num ", len(feature_column))
    data_local = []  # list of restaurants on specific page
    for col in feature_column:  # runs on every restaurant page
        try:
            pageid = col.attrs["data-customer"]
            print("page id "+pageid)
            url = "https://www.rest.co.il/rest/" + pageid
            feature_body = get_page_soup(url)
            name = get_name(feature_body)
            type = get_type(feature_body)
            page_attributes = get_page_attributes_sel(url, feature_body)
            num_of_reviews = get_number_of_reviews(feature_body)
            stars = get_stars(feature_body)
            geolocation = get_geolocation(feature_body)
            resturant = {
                'id'             : pageid,
                'name'           : name,
                'type'           : type,
                'stars'          : stars,
                'location'       : geolocation,
                'num_of_reviews' :  num_of_reviews
            }
            for att in page_attributes:
                resturant[att] = '1'    
            print(resturant['name'])
            data_local.append(resturant)
        except Exception as e:
            print("[extract_page_attributes] error: ", e)
    return data_local

def get_data_for_pages(num):
    page = get_page_soup("https://www.rest.co.il/restaurants/israel")
    data.extend(extract_page_attributes(page))  # adding the page lst of JSON to global data DF
    if(num == 1):
        return data
    for i in range(2,num + 1):
        print("page ", i)
        page = get_page_soup("https://www.rest.co.il/restaurants/israel/page-{}/".format(i))
        if(page is None):
            break
        data.extend(extract_page_attributes(page))
    return data

def score_equation(df, row):
    #ToDO: delete this function or fix to implment instead of labda
    """
    calculate score of rest
    x - c*(s/sqrt(n))
    x = stars
    c = 1.96
    n = number of reviews
    :param df:
    :return:

    x̄ = sample mean
    μ0 = population mean
    s = sample standard deviation
    n = sample size

    """
    x = df['stars'][row]
    c = 1.96
    n = df['num_of_reviews'][row]
    s = 1  # std of the stars

    return (x-(c * ( s / math.sqrt(n))))

def load_csv(file_name): #for testing
    return pd.read_csv(file_name, header=0, sep=',')

def fill_empty_binary_values(df):
    #df.loc[:, ~df.columns.isin(['id', 'name', 'stars', 'location', 'num_of_reviews'])] = df.loc[:, ~df.columns.isin(['id', 'name', 'stars', 'location', 'num_of_reviews'])].fillna(value=0)
    df.fillna(value=0, inplace=True)

def heat_map(df):
    corr = df.corr()  # heat map
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)  #
    plt.show()

def update_score(df):
    df['score'] = df.apply(lambda row: row.stars - (1.96 * (1 / math.sqrt(row.num_of_reviews))), axis=1)  # update score column
    df['score_normalized'] = df.apply(lambda row: (row.score - min(df.score)) / (max(df.score) - min(df.score)), axis=1)  # normalized = (x-min(x))/(max(x)-min(x))

def split_loc(df):
    """
    splits the location into lat and lon columns and updates the df
    :param df:
    :return:
    """
    lat = []
    lon = []

    location = df['location'].tolist()
    for i in range(len(location)):
        if location[i] == '0' or location[i] == 0:
            lat.append(0)
            lon.append(0)
        else:
            temp = location[i].replace(" ", "").replace("(", "").replace(")", "").split(",")  # .trim()
            lat.append(temp[0][:4])
            lon.append(temp[1][:4])

    df['lat'] = lat
    df['lon'] = lon
    return df

def type_to_int(df):
    """
    turns the categorial type to numeric
    :param df:
    :return:
    """
    # le = preprocessing.LabelEncoder()
    # list = df['type']
    # list = le.fit_transform(list)
    # df.insert(loc=3, column='type_numeric', value=list)


    # df.type = pd.Categorical(df.type)
    # df['type_numeric'] = df.type.cat.codes

    # lbl = LabelEncoder()
    # df['type_numeric'] = lbl.fit_transform(df['type'])
    return df

def heat_map(df):
    corr = df.corr()  # heat map
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)  #
    plt.show()


#make this work
# def geo_map(df):
#     """
#     creates a map with visualization of the scores of each restaurants
#     """
#     lat = df['lat'].values
#     lon = df['lon'].values
#     score = df['score_normalized'].values
#
#     fig = plt.figure(figsize=(8, 8))
#
#     m = Basemap(projection='lcc', resolution='h',
#                 width=0.5E6, height=0.5E6,
#                 lat_0=31.6, lon_0=34.88, )
#
#     m.shadedrelief()
#     m.drawcoastlines(color='gray')
#     m.drawcountries(color='gray')
#     m.drawstates(color='gray')
#
#
#
#     m.scatter(lon, lat, latlon=True, c = score,s = 15, cmap='Reds', alpha=0.3)  # c=np.log10(score)
#
#     plt.colorbar(label='score')
#     plt.clim(0, 1)
#
#     # Map (long, lat) to (x, y) for plotting
#     x, y = m(32, 34)
#     plt.plot(x, y, 'ok', markersize=2)
#     plt.text(x, y, ' scores', fontsize=12)
#     plt.show()

# remove this func
def show_rest_map(df, gpd=None):
    geo_df = pd.DataFrame(columns=['id', 'lat', 'lon', 'score'])

    geo_df['id'] = df['id']
    #geo_df['lat'] = df['location'].str.extract(r'(.*),')
    #geo_df['lon'] = df['location'].str.extract(r'(\w+(?: \w+)*)$')
    geo_df['lat'] = df['lat']
    geo_df['lon'] = df['lon']
    geo_df['score'] = df['score']

    geo_df = geo_df[geo_df.lat != '0']
    geo_df = geo_df[geo_df.lon != '0']

    a = 1

    geometry = [Point(xy) for xy in zip(geo_df['lat'], geo_df['lon'])]
    gdf = GeoDataFrame(df, geometry=geometry)

    # this is a simple map that goes with geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15)

def show_histograms(df):
    type_col = df['type_numeric'].to_list()
    values, counts = np.unique(type_col, return_counts=True)
    ind = np.argpartition(-counts, kth=10)[:10]

    for type in ind:  # show best 10 type graphs
        df_type = df.loc[df['type_numeric'] == type]
        name = (df_type.iloc[0]['type'])[::-1]
        x = df_type.index
        y = df_type['score_normalized'].to_list()

        ax = sns.displot(y, kde=True)
        ax.set(xlabel='score', ylabel='number of restaurants')
        ax.fig.suptitle(name)
        plt.show()



def show_boxplot(df):
    sns.set_theme(style="white", palette="pastel")
    ax = sns.boxplot(x=df["score"])
    plt.show()

# data = get_data_for_pages(400)
# df = pd.DataFrame.from_records(data)
# save_df_to_csv(df)

df = load_csv("Resturants Output/6kdata.csv")

fill_empty_binary_values(df)
df = df[df.num_of_reviews != 0]
df = df.loc[:, (df != 0).any(axis=0)] #Removes columns with zeros only
df = type_to_int(df)
df = split_loc(df)
update_score(df)
save_df_to_csv(df)
#print(df)


#EDA

#heat_map(df)
#show_rest_map(df)
#show_histograms(df)
#show_boxplot(df)


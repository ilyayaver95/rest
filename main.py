import math
from turtle import pd
import pandas as pd
import numpy as np
from attr import attributes
from bs4 import BeautifulSoup
from datetime import datetime
import requests
import urllib3
import os
from selenium.webdriver.chrome.options import Options
from geopy.geocoders import Nominatim
import geopy
from pathlib import Path  
from selenium import webdriver
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from collections import Counter


from scipy import stats
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
data = []

'''
# 
# final_box: func(stars + num_feedbacks)....
# '''

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
    driver_path = "C:\Program Files\chromedriver.exe"
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(driver_path, options=options)
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
    # return address.text  # take the address

def save_df_to_csv(df):
    """
    cerates csv based on the DF.
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
    # df = df.loc[:, ~df.columns.isin(['id', 'name', 'stars', 'location', 'num_of_reviews'])].fillna(value=0)
    df = df.fillna(value= 0)
    return df

def heat_map(df):
    corr = df.corr()  # heat map
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)  #
    plt.show()

def update_score(df1):
    df = df1
    save_df_to_csv(df)

    df['score'] = df.apply(lambda row: float(row.stars) - (1.96 * (1 / math.sqrt(int(row.num_of_reviews)))), axis=1)  # update score column
    df['score_normalized'] = df.apply(lambda row: (row.score - min(df.score)) / (max(df.score) - min(df.score)), axis=1)  # normalized = (x-min(x))/(max(x)-min(x))

    return df

def type_to_int(df1):
    """
    turns the categorial type to numeric
    :param df:
    :return:
    """
    df = df1
    le = preprocessing.LabelEncoder()

    list = []
    # df['type_numeric'] = df['type']
    # df['type_numeric']  = le.fit_transform(df['type_numeric'])

    list = df['type']
    list = le.fit_transform(list)
    df.insert(loc=3, column='type_numeric', value = list)

    return df
def gather_features(df1):
    """
    ges the df and combines columns with high correlation to one column
    :param df:
    :return:
    """

    return df

def trimm_correlated(df, threshold):
    # df_in = df.drop(columns = ['id', 'name', 'stars', 'location', 'score', 'score_normalized'])  # num_of_reviews
    df_in = df
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]
    return df_out

def get_redundant_pairs(df1):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    df = df1.iloc[:,5:-3]
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def hight_corr(df1, n=5):
        df = df1.iloc[:,5:-2]
        au_corr = df.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

def log_regression(df):
    # lrm = linear_model.LogisticRegression()
    # lrm.fit(df[:,5:-3], df["score_normalized "])
    #  test size = 25
    data = df.drop(columns = ['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'type', 'type_numeric', 'num_of_reviews']).copy()  # num_of_reviews
    score_normalized = df["score_normalized"]
    x_train, x_test, y_train, y_test = train_test_split(data, score_normalized , test_size=0.20, random_state=0)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)

    logisticRegr.predict(x_test)  # [0:10]

    score = logisticRegr.score(x_test, y_test)

    print("logistic regression score:")
    print(score)

    importance = logisticRegr.coef_[0]

    # summarize feature importance
    # for i, v in enumerate(importance):
    #
    #     print('Feature: %0d, Score: %.5f' % (i, v))




def knn_regression(df):
    data = df.drop(columns=['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'type', 'type_numeric',
                            'num_of_reviews']).copy()  # num_of_reviews
    score_normalized = df["score_normalized"]
    X_train, X_test, y_train, y_test = train_test_split(data, score_normalized, test_size=0.2, random_state=12)

    knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

    # Score
    score_knn = knn_model.score(X_test, y_test)
    print("knn score:")
    print(score_knn)

# data = get_data_for_pages(100)  # original
# df = pd.DataFrame.from_records(data)

# save_df_to_csv(df)

#
# print(df.columns)
# df = fill_empty_binary_values(df)  # fixed
# df = df[df.num_of_reviews != 0]
# df = update_score(df)
# df = type_to_int(df)
# save_df_to_csv(df)
# print(df)

df = load_csv(r"C:\Users\IlyaY\PycharmProjects\rest\Resturants Output\Rest df 22.May.2022 23-01-25_very big.csv")
df = df.loc[:, df.any()]  # remove column with all zeros
type_col = df['type_numeric'].to_list()

values, counts = np.unique(type_col, return_counts=True)  # finds the most common types of restaurant
ind = np.argpartition(-counts, kth=10)[:10]
# print(ind)

# for type in ind:    # show best 10 type graphs
#     df_type = df.loc[df['type_numeric'] == type]
#     name = (df_type.iloc[0]['type'])[::-1]
#     # print(df_type.columns)
#
#     x = df_type.index
#     y = df_type['score_normalized '].to_list()
#
#     ax = sns.displot(y, kde=True)
#     ax.set(xlabel='score', ylabel='number of restaurants')
#     print(name)
#     ax.fig.suptitle(name)
#     plt.show()

# heat_map(df)


print("Top Absolute Correlations")
#print(hight_corr(df, 3))

# print(df.corr().unstack().sort_values().drop_duplicates())
# test_df = trimm_correlated(df, 0.6)  # delete high correlated columns TODO: fix
test_df = df.drop(columns = ['גישה לתחבורה ציבורית', 'שירות הזמן שולחן' ,'נגישות לנכים' ,'אזור עישון', 'ציוד הגברה','ימי הולדת','ישיבות','אירועים קטנים', 'משלוחים']).copy()
# print('after trim')
#log_regression(test_df)

log_regression(df)
knn_regression(df)

heat_map(test_df)


# bistro = df.loc[df['type_by_numeric'] == 1]
# cafe = df.loc[df['type_by_numeric'] == 2]
# bistro = df.loc[df['type_by_numeric'] == 1]

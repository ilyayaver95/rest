import math
import os
from datetime import datetime
from pathlib import Path
from turtle import pd

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import urllib3
from bs4 import BeautifulSoup
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from scipy.stats import spearmanr
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from shapely.geometry import Point
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  # to standardize the features
from mpl_toolkits.basemap import Basemap
import geopandas as gpd

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

    return df1

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

def dim_reduce_PCA(df):


    data = df.drop(columns=['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'type', 'num_of_reviews']).copy()
    features = data.columns
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:, ['score_normalized']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=20)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents)

    print(principalDf)
    finalDf = pd.concat([principalDf, df[['score_normalized']]], axis=1)
    finalDf = pd.concat([finalDf, df[['lat','lon','type_numeric']]], axis=1)
    return finalDf

def show_rest_map(df):
    geo_df = pd.DataFrame(columns=['id','lat', 'lon', 'score'])
    location = df['location']

    geo_df['id'] = df['id']
    geo_df['lat'] = df['location'].str.extract(r'(.*),')
    geo_df['lon'] = df['location'].str.extract(r'(\w+(?: \w+)*)$')

    geometry = [Point(xy) for xy in zip(geo_df['lat'], geo_df['lon'])]
    gdf = GeoDataFrame(df, geometry=geometry)

    # this is a simple map that goes with geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15)

def split_loc(df):
    """
    splits the location into lat and lon columns and updates the df
    :param df:
    :return:
    """
    lat = []
    lon = []

    location = df['location']
    for i in range(len(location)):
        if (location[i] == '0'):
            lat.append(0)
            lon.append(0)
        else:
            temp = location[i].replace(" ", "").replace("(", "").replace(")", "").split(",")  # .trim()
            lat.append(temp[0][:4])
            lon.append(temp[1][:4])

    df['lat'] = lat
    df['lon'] = lon
    return df
def geo_map(df):
    """
    creates a map with visualization of the scores of each restaurants
    """
    lat = df['lat'].values
    lon = df['lon'].values
    score = df['score_normalized'].values

    fig = plt.figure(figsize=(8, 8))

    m = Basemap(projection='lcc', resolution='h',
                width=0.5E6, height=0.5E6,
                lat_0=31.6, lon_0=34.88, )

    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    m.drawstates(color='gray')



    m.scatter(lon, lat, latlon=True, c = score,s = 15, cmap='Reds', alpha=0.3)  # c=np.log10(score)

    plt.colorbar(label='score')
    plt.clim(0, 1)

    # Map (long, lat) to (x, y) for plotting
    x, y = m(32, 34)
    plt.plot(x, y, 'ok', markersize=2)
    plt.text(x, y, ' scores', fontsize=12)
    plt.show()


def lin_regression(df):

    # df = df.drop(columns = ['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'type', 'num_of_reviews']).copy()  # num_of_reviews
    data = df.drop(columns = ['score_normalized'])
    # data = df[['lat', 'lon']]  # num_of_reviews

    # data = dim_reduce_PCA(data)
    score_normalized = df["score_normalized"]

    x_train, x_test, y_train, y_test = train_test_split(data, score_normalized, test_size=0.1)
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    # clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print("lin_regression score:")
    print(score)

    print("Slope:", clf.coef_)
    print("Intercept:", clf.intercept_)
    print("R2:", clf.score(data, score_normalized))
    print("R2:", r2_score(score_normalized, clf.predict(data.values)))

    return clf

def lin_reg2(df1):
    df = df1

    data = df.drop(columns = ['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'type', 'num_of_reviews']).copy()  # num_of_reviews
    score_normalized = df["score_normalized"]

    # Train the linear regression model
    reg = LinearRegression()
    model = reg.fit(data, score_normalized)

    # Generate a prediction
    # example = t.transform(pd.DataFrame([{
    #     'par1': 2, 'par2': 0.33, 'par3': 'no', 'par4': 'red'
    # }]))
    # prediction = model.predict(example)
    reg_score = reg.score(data, score_normalized)
    print(reg_score)

    return model

def knn_regression(df):
    # df = pd.get_dummies(df)
    data = df.drop(columns=['id', 'name', 'stars', 'location', 'score', 'score_normalized', 'type',
                            'num_of_reviews']).copy()  # num_of_reviews
    # data = df.drop(columns=['score_normalized']).copy()  # num_of_reviews
    data = pd.get_dummies(data)
    # data = dim_reduce_PCA(data)

    score_normalized = df["score_normalized"]
    X_train, X_test, y_train, y_test = train_test_split(data, score_normalized, test_size=0.2, random_state=12)

    scaler = MinMaxScaler(feature_range=(0, 1))

    x_train_scaled = scaler.fit_transform(X_train)
    x_train = pd.DataFrame(x_train_scaled)

    x_test_scaled = scaler.fit_transform(X_test)
    x_test = pd.DataFrame(x_test_scaled)

    knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
    params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
    knn = neighbors.KNeighborsRegressor()

    model = GridSearchCV(knn, params, cv=5)
    model.fit(x_train, y_train)
    print(model.best_params_)

    # Score
    score_knn = model.score(x_test, y_test)
    print("knn score:")
    print(score_knn)

def drop_low_coef_features(df):
    data = df.drop(columns = ['id', 'name', 'stars', 'location', 'score', 'type', 'type_numeric', 'num_of_reviews']).copy()  # num_of_reviews

    targetVar = 'score_normalized'
    corr_threshold = 0.03

    corr = spearmanr(data)
    corrSeries = pd.Series(corr[0][:, 0],
                           index=data.columns)  # Series with column names and their correlation coefficients
    corrSeries = corrSeries[(corrSeries.index != targetVar) & (corrSeries > corr_threshold)]  # apply the threshold

    vars_to_keep = list(corrSeries.index.values)  # list of variables to keep
    vars_to_keep.append(targetVar)  # add the target variable back in
    data2 = data[vars_to_keep]

    print(data2.columns)

    return data2

def check_example(model, example):

    prediction = model.predict(example)

    print("the predicted score is:")
    print(prediction)

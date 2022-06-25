# rest  25.26.22  @ ilya and oren

import numpy as np
import urllib3

from Utils.utils import get_page_attributes_sel
from Utils.utils import split_loc
from Utils.utils import load_csv
from Utils.utils import dim_reduce_PCA
from Utils.utils import lin_regression
from Utils.utils import knn_regression

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

df = load_csv(r"C:\Users\IlyaY\PycharmProjects\rest\Resturants Output\Rest df 22.May.2022 23-01-25_very big.csv")

df = df.loc[:, df.any()]  # remove columns with all zeros  todo: add to crawling
# type_col = df['type_numeric'].to_list()  # turn categorial column into numeric
#
# values, counts = np.unique(type_col, return_counts=True)  # finds the most common types of restaurant
# ind = np.argpartition(-counts, kth=10)[:10]  # take the 10 most frequent restaurant type
df = split_loc(df)

# ----visualizations---
# heat_map(df)
# geo_map(df)

test_df = df.drop(columns = ['גישה לתחבורה ציבורית', 'שירות הזמן שולחן' ,'נגישות לנכים' ,'אזור עישון', 'ציוד הגברה','ימי הולדת','ישיבות','אירועים קטנים', 'משלוחים']).copy()


df1 = dim_reduce_PCA(df)  # reduce the dimention of the feature matrix
model = lin_regression(df1)
# model = lin_reg2(df)

# model = lin_regression(df)

knn_regression(df)



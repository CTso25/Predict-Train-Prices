import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing

df_chunk = pd.read_csv("input/renfe.csv", chunksize=1000000)

chunk_list = []  # append each chunk df here

# Each chunk is in df format
for chunk in df_chunk:
    # perform data filtering
    # chunk_filter = chunk_preprocessing(chunk)

    # Once the data filtering is done, append the chunk to list
    chunk_list.append(chunk)

# concat the list into dataframe
renfe_df= pd.concat(chunk_list)

# print("Range: ")
# print("Min: " + renfe_df['start_date'].min())
#
# print("Max: " + renfe_df['start_date'].max())

# ponferrada_routes = renfe_df[(renfe_df['origin'] == 'PONFERRADA') | (renfe_df['destination'] == 'PONFERRADA')]
# print(ponferrada_routes['start_date'].min())
# print(ponferrada_routes['start_date'].max())

sevilla_routes = renfe_df[(renfe_df['origin'] == 'SEVILLA') | (renfe_df['destination'] == 'SEVILLA')]
print(sevilla_routes['train_type'].unique())
print(sevilla_routes['train_class'].unique())
print(sevilla_routes['fare'].unique())



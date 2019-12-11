# import things

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

renfe_nccd = pd.read_csv("input/not_encoded_clean_data.csv")
renfe_clean = pd.read_csv("input/cleaned_data.csv")
renfe_cleaned = renfe_clean.drop(columns = ['Unnamed: 0', 'insert_date', 'start_date', 'end_date'])

# renfe_cleaned = renfe_cleaned.drop(renfe_cleaned.index[renfe_cleaned['train_class']=='Cama G. Clase'].values)
# renfe_cleaned = renfe_cleaned.drop(renfe_cleaned.index[renfe_cleaned['fare']=='Individual Sleeper-Flexible'].values)
# renfe_cleaned = renfe_cleaned.drop(renfe_cleaned.index[renfe_cleaned['fare']=='Adulto Ida'].values)
# renfe_cleaned = renfe_cleaned.drop(renfe_cleaned.index[renfe_cleaned['origin']=='MADRID'].values)
# renfe_cleaned = renfe_cleaned.drop(renfe_cleaned.index[renfe_cleaned['destination']=='MADRID'].values)

# dfEdgeArtistMuseum = dfEdgeArtistMuseum.drop(dfEdgeArtistMuseum.index[dfEdgeArtistMuseum['museum_id']==dvalue].values)

renfe_cleaned = renfe_cleaned[renfe_cleaned.columns.values].astype(float)

# # is overnight
# fig,ax = plt.subplots(figsize=(20,6))
# ax = sns.countplot(renfe_nccd['is_overnight'])
# plt.xlabel("is overnight")
# plt.xticks([0, 1], ['No', 'Yes'])
# plt.show()
#
# fig,ax = plt.subplots(figsize=(20,6))
# ax = sns.boxplot(x='is_overnight', y='price', data=renfe_nccd)
# plt.xlabel("is overnight")
# plt.xticks([0, 1], ['No', 'Yes'])
# plt.show()

# # train type
# fig,ax = plt.subplots(figsize=(20,6))
# ax = sns.countplot(renfe_nccd['train_type'])
# plt.show()

# fig,ax = plt.subplots(figsize=(20,6))
# ax = sns.boxplot(x='train_type', y='price', data=renfe_nccd)
# plt.show()

# # train class
# fig,ax = plt.subplots(figsize=(20,6))
# ax = sns.countplot(renfe_nccd['train_class'])
# plt.show()

class_list = renfe_nccd['train_class']
y_list = renfe_nccd['price']
x = []
y = []
for i in range(len(class_list)):
    if class_list[i] != 'Cama G. Clase':
        x.append(class_list[i])
        y.append(y_list[i])

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x=x, y=y)
plt.show()


# # fare
# fig,ax = plt.subplots(figsize=(20,6))
# ax = sns.countplot(renfe_nccd['fare'])
# plt.show()

fare_list = renfe_nccd['fare']
y_list = renfe_nccd['price']
x = []
y = []
for i in range(len(fare_list)):
    if (fare_list[i] != 'Individual Sleeper-Flexible' and fare_list[i] != 'Adulto Ida'):
        x.append(fare_list[i])
        y.append(y_list[i])

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x=x, y=y)
plt.show()

# day of the week
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['dotw'])
plt.xlabel("day of the week")
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='dotw', y='price', data=renfe_nccd)
plt.xlabel("day of the week")
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun'])
plt.show()

# date
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['date'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='date', y='price', data=renfe_nccd)
plt.show()

# month
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['month'])
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='month', y='price', data=renfe_nccd)
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct'])
plt.show()

# days to holiday
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['days_to_holiday'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='days_to_holiday', y='price', data=renfe_nccd)
plt.show()

# days from holiday
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['days_from_holiday'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='days_from_holiday', y='price', data=renfe_nccd)
plt.show()

# days to trip
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['days_to_trip'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='days_to_trip', y='price', data=renfe_nccd)
plt.show()

# correlation plot

f,ax = plt.subplots(figsize=(30, 30))
sns.heatmap(renfe_cleaned.corr(), annot=False, cmap="Blues", linewidths=.5, fmt='.2f', ax=ax)
plt.show()


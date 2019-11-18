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

# is overnight
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['is_overnight'])
plt.xlabel("is overnight")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='is_overnight', y='price', data=renfe_nccd)
plt.xlabel("is overnight")
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# train type
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['train_type'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='train_type', y='price', data=renfe_nccd)
plt.show()

# train class
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['train_class'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='train_class', y='price', data=renfe_nccd)
plt.show()

# fare
fig,ax = plt.subplots(figsize=(20,6))
ax = sns.countplot(renfe_nccd['fare'])
plt.show()

fig,ax = plt.subplots(figsize=(20,6))
ax = sns.boxplot(x='fare', y='price', data=renfe_nccd)
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
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(renfe_nccd, annot=False, cmap = "Blues", linewidths=.5, fmt='.2f', ax=ax)
plt.show()

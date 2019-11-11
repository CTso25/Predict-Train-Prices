# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 18:53:16 2019

@author: Chris
"""

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

renfedata = pd.read_csv("input/renfe.csv")

# drop nan values to reduce processing time
renfedata = renfedata.dropna()

# add in population

# check unique cities

renfedata.origin.unique()
renfedata.destination.unique()

# renfedata.destinationpop.unique()
# Madrid Population 2018 - 6.55 million
# Barcelona Population 2018 - 5.515 million
# Sevilla Populaiton 2018 - 1.945 million
# Valencia Population 2018 - 2.531 million
# Ponferrada Population 2018 - 65,239

# process dates
renfedata['insert_date'] = pd.to_datetime(renfedata['insert_date'])
renfedata['start_date'] = pd.to_datetime(renfedata['start_date'])
renfedata['end_date'] = pd.to_datetime(renfedata['end_date'])

renfedata['travel_mins'] = renfedata['end_date'] - renfedata['start_date']
renfedata['travel_mins'] = renfedata['travel_mins']/np.timedelta64(1, 'm')

# set new columns
renfedata['originpop'] = 0
renfedata['destinationpop'] = 0


# helper functions for conditional origin/city populations
def originpop(data):
    if data['origin'] == 'MADRID':
        return 6550000
    elif data['origin'] == 'BARCELONA':
        return 5515000
    elif data['origin'] == 'SEVILLA':
        return 1945000
    elif data['origin'] == 'VALENCIA':
        return 2531000
    elif data['origin'] == 'PONFERRADA':
        return 65239


def destpop(data):
    if data['destination'] == 'MADRID':
        return 6550000
    elif data['destination'] == 'BARCELONA':
        return 5515000
    elif data['destination'] == 'SEVILLA':
        return 1945000
    elif data['destination'] == 'VALENCIA':
        return 2531000
    elif data['destination'] == 'PONFERRADA':
        return 65239


renfedata['originpop'] = renfedata.apply(originpop, axis=1)
renfedata['destinationpop'] = renfedata.apply(destpop, axis=1)

# distances taken from here: https://www.trenes.com/

# 505 km from Madrid to Barcelona
# 390 km from Madrid to Sevilla
# 302 km from Madrid to Valencia
# 338 km from Madrid to Ponferrada
# 829 km from Barcelona to Sevilla
# 303 km from Barcelona to Valencia
# 737 km from Barcelona to Ponferrada
# 539 km from Sevilla to Valencia
# 728 km from Sevilla to Ponferrada
# 641 km from Valencia to Ponferrada

# check unique routes
renfe_unique_o_d = renfedata.groupby(['origin', 'destination']).size().reset_index(name='Freq')
print(renfe_unique_o_d)


# helper function for distances between cities
def distmeasure(data):
    if data['destination'] == 'MADRID' and data['origin'] == 'BARCELONA':
        return 505
    elif data['destination'] == 'BARCELONA' and data['origin'] == 'MADRID':
        return 505
    elif data['destination'] == 'MADRID' and data['origin'] == 'SEVILLA':
        return 390
    elif data['destination'] == 'SEVILLA' and data['origin'] == 'MADRID':
        return 390
    elif data['destination'] == 'MADRID' and data['origin'] == 'VALENCIA':
        return 302
    elif data['destination'] == 'VALENCIA' and data['origin'] == 'MADRID':
        return 302
    elif data['destination'] == 'MADRID' and data['origin'] == 'PONFERRADA':
        return 338
    elif data['destination'] == 'PONFERRADA' and data['origin'] == 'MADRID':
        return 338

renfedata['distance'] = renfedata.apply(distmeasure, axis=1)

# convert start_date to month, date, day of the week, hour, minute
renfedata['month'] = renfedata['start_date'].dt.month
renfedata['date'] = renfedata['start_date'].dt.day
renfedata['dotw'] = renfedata['start_date'].dt.day_name
renfedata['hour'] = renfedata['start_date'].dt.hour
renfedata['minute'] = renfedata['start_date'].dt.minute

# helper function to determine if train is "overnight", or ends on different day than it starts
def isOvernight(data):
    if pd.to_datetime(data['start_date']).day != pd.to_datetime(data['end_date']).day:
        return 1
    else:
        return 0


renfedata['is_overnight'] = renfedata.apply(isOvernight, axis=1)


# consolidate the fare type into the
def fareType(data):
    if data['fare'] == 'Promo':
        return 'Promo'
    elif data['fare'] == 'Flexible':
        return 'Flexible'
    elif data['fare'] == 'Adulto ida':
        return 'Adulto Ida'
    elif data['fare'] == 'Promo +':
        return 'Promo +'
    elif data['fare'] == 'Grupos Ida':
        return 'Grupos Ida'
    elif data['fare'] == 'Double Sleeper-Flexible':
        return 'Flexible'
    elif data['fare'] == 'COD.PROMOCIONAL':
        return 'Promo'
    elif data['fare'] == 'Mesa':
        return 'Table'
    elif data['fare'] == 'Individual-Flexible':
        return 'Individual Sleeper-Flexible'
    else:
        return np.nan


# consolidate the train classes
def trainClasses(data):
    if data['train_class'] == 'TuristaSólo plaza H':
        return np.nan
    elif data['train_class'] == 'PreferenteSólo plaza H':
        return np.nan
    elif data['train_class'] == 'Turista PlusSólo plaza H':
        return np.nan
    else:
        return data['train_class']


renfedata['fare'] = renfedata.apply(fareType, axis=1)
renfedata['train_class'] = renfedata.apply(trainClasses, axis=1)

renfedata = renfedata.dropna()

renfedata = pd.get_dummies(renfedata, columns=['origin', 'destination', 'day_name', 'train_type', 'fare', 'train_class'])

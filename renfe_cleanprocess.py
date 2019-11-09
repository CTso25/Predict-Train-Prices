# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:02:21 2019

@author: Chris
"""
#import things

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

renfedata = pd.read_csv("input/renfe.csv")

#add in population

#check unique cities

renfedata.origin.unique()
renfedata.destination.unique()
#renfedata.destinationpop.unique()
#Madrid Population 2018 - 6.55 million
#Barcelona Population 2018 - 5.515 million
#Sevilla Populaiton 2018 - 1.945 million
#Valencia Population 2018 - 2.531 million
#Ponferrada Population 2018 - 65,239 

#set new columns
renfedata['originpop'] = 0
renfedata['destinationpop'] = 0
        
#helper functions for conditional origin/city populations
def originpop(data):
    if data['origin'] == 'MADRID':
        return 6550000
    if data['origin'] == 'BARCELONA':
        return 5515000
    if data['origin'] == 'SEVILLA':
        return 1945000
    if data['origin'] == 'VALENCIA':
        return 2531000
    if data['origin'] == 'PONFERRADA':
        return 65239

def destpop(data):
     if data['destination'] == 'MADRID':
         return 6550000
     if data['destination'] == 'BARCELONA':
        return 5515000
     if data['destination'] == 'SEVILLA':
        return 1945000
     if data['destination'] == 'VALENCIA':
        return 2531000
     if data['destination'] == 'PONFERRADA':
        return 65239   

renfedata['originpop'] = renfedata.apply(originpop, axis = 1)
renfedata['destinationpop'] = renfedata.apply(destpop, axis = 1)

#Euclidian Distance
#based on driving distance
#Madrid to Barcelona
#Madrid to Sevilla
#Madrid to Valencia
#Madrid to Ponferrada

#this is a  test 





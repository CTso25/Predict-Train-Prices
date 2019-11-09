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

#process dates
renfedata['insert_date'] = pd.to_datetime(renfedata['insert_date'])
renfedata['start_date'] = pd.to_datetime(renfedata['start_date'])
renfedata['end_date'] = pd.to_datetime(renfedata['end_date'])

renfedata['travel_mins'] = renfedata['end_date'] - renfedata['start_date']
renfedata['travel_mins'] = renfedata['travel_mins']/np.timedelta64(1,'m')

#set new columns
renfedata['originpop'] = 0
renfedata['destinationpop'] = 0
        
#helper functions for conditional origin/city populations
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

renfedata['originpop'] = renfedata.apply(originpop, axis = 1)
renfedata['destinationpop'] = renfedata.apply(destpop, axis = 1)

#distances taken from here: https://www.trenes.com/

#505 km from Madrid to Barcelona
#390 km from Madrid to Sevilla
#302 km from Madrid to Valencia
#338 km from Madrid to Ponferrada
#829 km from Barcelona to Sevilla
#303 km from Barcelona to Valencia
#737 km from Barcelona to Ponferrada
#539 km from Sevilla to Valencia
#728 km from Sevilla to Ponferrada
#641 km from Valencia to Ponferrada

#check unique routss
renfe_unique_o_d = renfedata.groupby(['origin', 'destination']).size().reset_index(name = 'Freq')
print (renfe_unique_o_d)

#helper function for distances between cities

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
    elif data['destination'] == 'PONDERRADA' and data['origin'] == 'MADRID':  
        return 338
#    elif data['destination'] == 'BARCELONA' and data['origin'] == 'SEVILLA':  
#        return 829
#    elif data['destination'] == 'SEVILLA' and data['origin'] == 'BARCELONA':
#        return 829    
#    elif data['destination'] == 'BARCELONA' and data['origin'] == 'VALENCIA':  
#        return 303
#    elif data['destination'] == 'VALENCIA' and data['origin'] == 'BARCELONA':
#        return 303    
#    elif data['destination'] == 'BARCELONA' and data['origin'] == 'PONFERRADA':  
#        return 737
#    elif data['destination'] == 'PONFERRADA' and data['origin'] == 'BARCELONA':
#        return 737  
#    elif data['destination'] == 'SEVILLA' and data['origin'] == 'VALENCIA':  
#        return 539
#    elif data['destination'] == 'VALENCIA' and data['origin'] == 'SEVILLA':
#        return 539
#    elif data['destination'] == 'SEVILLA' and data['origin'] == 'PONFERRADA':
#        return 728
#    elif data['destination'] == 'PONFERRADA' and data['origin'] == 'SEVILLA':
#        return 728
#    elif data['destination'] == 'VALENCIA' and data['origin'] == 'PONFERRADA':
#        return 641
#    elif data['destination'] == 'PONFERRADA' and data['origin'] == 'VALENCIA':
#        return 641
        
renfedata['distance'] = renfedata.apply(distmeasure, axis = 1)






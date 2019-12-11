import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# read in data that was engineered and preprocessed
renfedata = pd.read_csv("input/cleaned_data.csv")

# sample data for easy of modeling and quicker training
renfedata = renfedata.sample(n=100000, random_state=0)

# re-create the original renfe dataframe cleaned without engineered features (i.e - days to, populations)
original_renfe_df = renfedata[['insert_date', 'price', 'month', 'date', 'hour', 'minute', 'origin_BARCELONA',
                               'origin_MADRID', 'origin_PONFERRADA', 'origin_SEVILLA', 'origin_VALENCIA',
                               'destination_BARCELONA', 'destination_MADRID', 'destination_PONFERRADA',
                               'destination_SEVILLA', 'destination_VALENCIA','dotw_0', 'dotw_1',
                               'dotw_2', 'dotw_3', 'dotw_4', 'dotw_5', 'dotw_6', 'train_type_ALVIA',
                               'train_type_AV City', 'train_type_AVE', 'train_type_AVE-LD',
                               'train_type_AVE-MD', 'train_type_AVE-TGV', 'train_type_INTERCITY',
                               'train_type_LD', 'train_type_LD-MD', 'train_type_MD',
                               'train_type_MD-AVE', 'train_type_MD-LD', 'train_type_R. EXPRES',
                               'train_type_REGIONAL', 'train_type_TRENHOTEL', 'fare_Adulto Ida',
                               'fare_Flexible', 'fare_Grupos Ida', 'fare_Individual Sleeper-Flexible',
                               'fare_Promo', 'fare_Promo +', 'fare_Table', 'train_class_Cama G. Clase',
                               'train_class_Cama Turista', 'train_class_Preferente',
                               'train_class_Turista', 'train_class_Turista Plus',
                               'train_class_Turista con enlace']]

def prepare_data():
    # Define explanatory variables (features) and response variable (price)
    features = original_renfe_df.drop(columns=['price'], axis=1)
    response = original_renfe_df[['price']]

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, response, train_size=0.80)

    # verify that training and testing are separated and their ranges
    print('Train data(range):')
    print(X_train['insert_date'].min())
    print(X_train['insert_date'].max())
    print('Test data(range):')
    print(X_test['insert_date'].min())
    print(X_test['insert_date'].max())

    # remove time stamps
    X_train = X_train.drop(columns=['insert_date'], axis=1)
    X_test = X_test.drop(columns=['insert_date'], axis=1)

    # standardize X_train and X_test
    minMax_scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train_scaled = minMax_scaler.transform(X_train)
    X_test_scaled = minMax_scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# function to return all features in a list
def get_features():
    features = original_renfe_df.drop(columns=['price', 'insert_date'], axis=1)
    return list(features.columns)

# function to get upper and lower bounds of the response variable values based on given percent
def get_bounds(y, percent):
    float_percent = percent/100
    upper_bound = y + (y * float_percent)
    lower_bound = y - (y * float_percent)
    lower_bound.columns = ['lower_bound']
    upper_bound.columns = ['upper_bound']
    bounds = lower_bound.join(upper_bound)
    return bounds

# function to compute accuracy scores for predictions based on actual upper/lower bound limits
def get_interval_accuracy_score(bounds, y):
    preds_acc = []
    for i in range(len(y)):
        if (bounds.iloc[i].lower_bound <= y[i] <= bounds.iloc[i].upper_bound):
            preds_acc.append(1)
        else:
            preds_acc.append(0)
    return np.mean(preds_acc)
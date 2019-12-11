import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

# read in original data set provided from Kaggle (7 million records)
renfedata = pd.read_csv("input/renfe.csv")

# check unique origin/destinations of data set
renfedata.origin.unique()
renfedata.destination.unique()

# process dates
renfedata['insert_date'] = pd.to_datetime(renfedata['insert_date'])
renfedata['start_date'] = pd.to_datetime(renfedata['start_date'])
renfedata['end_date'] = pd.to_datetime(renfedata['end_date'])

# create feature for travel time
renfedata['travel_mins'] = renfedata['end_date'] - renfedata['start_date']
renfedata['travel_mins'] = renfedata['travel_mins']/np.timedelta64(1, 'm')

# set new columns for population size feature and initialize to 0
renfedata['originpop'] = 0
renfedata['destinationpop'] = 0


# helper function for to update populations for origin
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


# helper function for to update populations for origin
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


# apply population helper function update newly created features highlighting city population
renfedata['originpop'] = renfedata.apply(originpop, axis=1)
renfedata['destinationpop'] = renfedata.apply(destpop, axis=1)

# read in holidays.csv to dataframe
holidays_df = pd.read_csv('datasets/holidays.csv')

# create dictionary to store cities as hash keys
holidays_dict = {
    "MADRID": [],
    "BARCELONA": [],
    "VALENCIA": [],
    "SEVILLA": [],
    "PONFERRADA": []
}

# add all holiday dates to dictionary for respective city
for index, row in holidays_df.iterrows():
    holidays_dict[holidays_df['rel_city'][index]].append(pd.to_datetime(holidays_df['date'][index]))


# helper function returns number of days away until next holiday/festival based on given date
def daysToHoliday(items_mad, items_other, ticket_date):
    nearest_mad_hol_date = min(items_mad, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days >= 0 else float('inf'))
    nearest_otr_hol_date = min(items_other, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days >= 0 else float('inf'))

    # check to see if both cities have at least one holiday date in the future
    # if not, only return the maximum date between the both i.e -  Madrid's next holiday
    if nearest_mad_hol_date.date() >= ticket_date.date() and nearest_otr_hol_date.date() >= ticket_date.date():
        nearest_date = min(nearest_mad_hol_date, nearest_otr_hol_date)
    else:
        nearest_date = max(nearest_mad_hol_date, nearest_otr_hol_date)

    return (nearest_date.date() - ticket_date.date()).days


# helper function returns number of days that have passed since last holiday/festival based on given date
def daysFromHoliday(items_mad, items_other, ticket_date):
    nearest_mad_hol_date = max(items_mad, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days <= 0 else float('-inf'))
    nearest_otr_hol_date = max(items_other, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days <= 0 else float('-inf'))

    # check to see if both cities have at least one holiday date in the past
    # if not, only return the maximum date between the both i.e -  Madrid's last holiday
    if nearest_mad_hol_date.date() <= ticket_date.date() and nearest_otr_hol_date.date() <= ticket_date.date():
        nearest_date = max(nearest_mad_hol_date, nearest_otr_hol_date)
    else:
        nearest_date = min(nearest_mad_hol_date, nearest_otr_hol_date)

    return abs((nearest_date.date() - ticket_date.date()).days)


# function to compute days to holiday on full data frame
def computeDaysToHoliday(data):
    if data['origin'] == 'MADRID':
        city_to_check = data['destination']
    else:
        city_to_check = data['origin']

    # extract list of holiday dates for MADRID and other CITY
    madrid_dates = holidays_dict.get('MADRID')
    local_city_dates = holidays_dict.get(city_to_check)

    # compute the minimum number of days away to nearest holiday for both and return the minimum of both
    return daysToHoliday(madrid_dates, local_city_dates, pd.to_datetime(data['start_date']))


# function to compute days from holiday on full data frame
def computeDaysFromHoliday(data):
    if data['origin'] == 'MADRID':
        city_to_check = data['destination']
    else:
        city_to_check = data['origin']

    # extract list of holiday dates for MADRID and other CITY
    madrid_dates = holidays_dict.get('MADRID')
    local_city_dates = holidays_dict.get(city_to_check)

    # compute the minimum number of days away to nearest holiday for both and return the minimum of both
    return daysFromHoliday(madrid_dates, local_city_dates, pd.to_datetime(data['start_date']))


# apply and create two new columns in data frame
renfedata['days_to_holiday'] = renfedata.apply(computeDaysToHoliday, axis=1)
renfedata['days_from_holiday'] = renfedata.apply(computeDaysFromHoliday, axis=1)


# To compute distances between origin/destination, distances taken from here: https://www.trenes.com/
# Distances Used:
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

# helper function to update distances between cities
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


# create distance feature to compute distance between origin & destination
renfedata['distance'] = renfedata.apply(distmeasure, axis=1)


# convert start_date to month, date, day of the week, hour, minute
renfedata['month'] = pd.to_datetime(renfedata['start_date']).dt.month
renfedata['date'] = pd.to_datetime(renfedata['start_date']).dt.day
renfedata['dotw'] = pd.to_datetime(renfedata['start_date']).dt.weekday
renfedata['hour'] = pd.to_datetime(renfedata['start_date']).dt.hour
renfedata['minute'] = pd.to_datetime(renfedata['start_date']).dt.minute


# helper function to determine if train is "overnight", or ends on different day than it starts
def isOvernight(data):
    if pd.to_datetime(data['start_date']).day != pd.to_datetime(data['end_date']).day:
        return 1
    else:
        return 0


# create feature to flag overnight trips
renfedata['is_overnight'] = renfedata.apply(isOvernight, axis=1)


# helper function to consolidate related fare types
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


# helper function to consolidate the train classes
def trainClasses(data):
    if data['train_class'] == 'TuristaSólo plaza H':
        return np.nan
    elif data['train_class'] == 'PreferenteSólo plaza H':
        return np.nan
    elif data['train_class'] == 'Turista PlusSólo plaza H':
        return np.nan
    else:
        return data['train_class']


# helper function to remove negative days to trips or illogical observations
def daystoTrip(data):
    if data['days_to_trip'] < 0:
        return np.nan
    else:
        return data['days_to_trip']


# update fare and train class features based on logic in helper functions above
renfedata['fare'] = renfedata.apply(fareType, axis=1)
renfedata['train_class'] = renfedata.apply(trainClasses, axis=1)

# create features for difference inn days away from trip as of scrape date
renfedata['days_to_trip'] = (renfedata['start_date'].dt.date - renfedata['insert_date'].dt.date).dt.days
renfedata['days_to_trip'] = renfedata.apply(daystoTrip, axis=1)

# drop nan values in remaining data set
renfedata = renfedata.dropna()

# save cleaned data set without categorical encoding for use in EDA purposes
renfedata.to_csv("input/not_encoded_clean_data.csv")

# perform one-hot encoding on categorical features
renfedata = pd.get_dummies(renfedata, columns=['origin', 'destination', 'dotw', 'train_type', 'fare', 'train_class'])

# save cleaned data set with categorical encoding for use in modeling purposes
renfedata.to_csv("input/cleaned_data.csv")

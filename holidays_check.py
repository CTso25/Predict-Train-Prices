import time
from datetime import datetime
import pandas as pd

# Read in dataset and use sample for testing purposes
renfedata = pd.read_csv("input/renfe.csv")
renfedata = renfedata.sample(n=50000, random_state=1)

# Read in holidays.csv to dataframe
holidays_df = pd.read_csv('datasets/holidays.csv')

# Create dictionary to store cities as hash keys
holidays_dict = {
    "MADRID": [],
    "BARCELONA": [],
    "VALENCIA": [],
    "SEVILLA": [],
    "PONFERRADA": []
}

# Add all holiday dates to dictionary for respective city
for index, row in holidays_df.iterrows():
    holidays_dict[holidays_df['rel_city'][index]].append(pd.to_datetime(holidays_df['date'][index]))

# # Test Cases [to eventually delete]
# ex_date = datetime.strptime('2019-06-09 05:50:00', '%Y-%m-%d %H:%M:%S')
# ex_date2 = datetime.strptime('2019-06-09 00:00:00', '%Y-%m-%d %H:%M:%S')
#
# # print((ex_date2.date() - ex_date.date()).days)
# # print(ex_date2.date())
# # print((ex_date2 - ex_date))
# date_list1 = holidays_dict.get('BARCELONA')
# date_list2 = holidays_dict.get('MADRID')

# Helper function returns number of days away until next holiday/festival based on given date
def daysToHoliday(items_mad, items_other, ticket_date):
    nearest_mad_hol_date = min(items_mad, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days >= 0 else float('inf'))
    nearest_otr_hol_date = min(items_other, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days >= 0 else float('inf'))

    # Check to see if both cities have at least one holiday date in the future
    # If not, only return the maximum date between the both i.e -  Madrid's next holiday
    if nearest_mad_hol_date.date() >= ticket_date.date() and nearest_otr_hol_date.date() >= ticket_date.date():
        nearest_date = min(nearest_mad_hol_date, nearest_otr_hol_date)
    else:
        nearest_date = max(nearest_mad_hol_date, nearest_otr_hol_date)

    return (nearest_date.date() - ticket_date.date()).days

# Helper function returns number of days that have passed since last holiday/festival based on given date
def daysFromHoliday(items_mad, items_other, ticket_date):
    nearest_mad_hol_date = max(items_mad, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days <= 0 else float('-inf'))
    nearest_otr_hol_date = max(items_other, key=lambda x: (x.date() - ticket_date.date()).days if (x.date() - ticket_date.date()).days <= 0 else float('-inf'))

    # Check to see if both cities have at least one holiday date in the past
    # If not, only return the maximum date between the both i.e -  Madrid's last holiday
    if nearest_mad_hol_date.date() <= ticket_date.date() and nearest_otr_hol_date.date() <= ticket_date.date():
        nearest_date = max(nearest_mad_hol_date, nearest_otr_hol_date)
    else:
        nearest_date = min(nearest_mad_hol_date, nearest_otr_hol_date)

    return abs((nearest_date.date() - ticket_date.date()).days)

# Function to compute days to holiday on full data frame
def computeDaysToHoliday(data):
    if data['origin'] == 'MADRID':
        city_to_check = data['destination']
    else:
        city_to_check = data['origin']

    # Extract list of holiday dates for MADRID and other CITY
    madrid_dates = holidays_dict.get('MADRID')
    local_city_dates = holidays_dict.get(city_to_check)

    # Compute the minimum number of days away to nearest holiday for both and return the minimum of both
    return daysToHoliday(madrid_dates, local_city_dates, pd.to_datetime(data['start_date']))

# Function to compute days from holiday on full data frame
def computeDaysFromHoliday(data):
    if data['origin'] == 'MADRID':
        city_to_check = data['destination']
    else:
        city_to_check = data['origin']

    # Extract list of holiday dates for MADRID and other CITY
    madrid_dates = holidays_dict.get('MADRID')
    local_city_dates = holidays_dict.get(city_to_check)

    # Compute the minimum number of days away to nearest holiday for both and return the minimum of both
    return daysFromHoliday(madrid_dates, local_city_dates, pd.to_datetime(data['start_date']))

# Apply and create two new columns in data frame
renfedata['days_to_holiday'] = renfedata.apply(computeDaysToHoliday, axis=1)
renfedata['days_from_holiday'] = renfedata.apply(computeDaysFromHoliday, axis=1)

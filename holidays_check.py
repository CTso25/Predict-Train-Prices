import time
from datetime import datetime
import pandas as pd

# Read
renfedata = pd.read_csv("input/renfe.csv")

renfedata_sample = renfedata.sample(n=50000, random_state=1)

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


ex_date = datetime.strptime('2019-05-14 05:50:00', '%Y-%m-%d %H:%M:%S')


date_list = holidays_dict.get('SEVILLA')

print(renfedata_sample.head())
# Function returns number of days away until next holiday/festival based on given date
def daysToHoliday(items, date):
    closest_date_in_future = min(items, key=lambda x: (x.day-date.day) if (x.day-date.day) >= 0 else float('inf'))
    return closest_date_in_future.day - date.day

# Function returns number of days that have passed since last holiday/festival based on given date
def daysFromHoliday(items, date):
    closest_date_in_past = max(items, key=lambda x: (x.day-date.day) if (x.day-date.day) <= 0 else float('-inf'))
    return abs(closest_date_in_past.day - date.day)

print(daysToHoliday(date_list, ex_date))
print(daysFromHoliday(date_list, ex_date))


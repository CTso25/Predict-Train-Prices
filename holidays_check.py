import csv

reader = csv.DictReader(open('datasets/holidays.csv'))

# Create dictionary to store holiday dates
holidays_dict = {}
for row in reader:
    for column, value in row.items():
        holidays_dict.setdefault(column, []).append(value)
print(holidays_dict)

#
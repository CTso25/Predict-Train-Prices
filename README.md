 # SML Project: Predicting Renfe Train Prices in Spain

Input data is not include in repository due to memory storage costs but can be downloaded in csv form from Kaggle
https://www.kaggle.com/thegurusteam/spanish-high-speed-rail-system-ticket-pricing and added into an 'input/' directory to easily read in

### Reproduceability Steps
The following steps must be performed in order to properly reproduce and generate the correct model results

__Feature Engineering__

_run feature_engineering.py to import dataset, create all featured engineered, 
and clean data (handle missing and incorrect data)_

__Model Preparation & Train/Test Splits__

model_prep_withFE.py is used to load in feature engineered + reduced dataset 

model_prep_withoutFE.py is used to load in data set without feature engineering 

__Models__

_run final_models.ipynb for final models used:_
* Linear Regression
* Random Forest
* Gradient Boosting (sklearn)
* XG Boost
* Cat Boost
* Neural Network (Feed-Forward)

### Additional Files in Repository

eda.py = exploratory data analysis performed

feature_selection.ipynb = feature selection procedures performed to reduce and eliminate non-needed features in data set

datasets = directory that includes csv file of all holidays in Spain used for create days to/from holiday features

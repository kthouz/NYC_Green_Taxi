# Use this code to predict the percentage tip expected after a trip in NYC green taxi
# The code is a predictive model that was built and trained on top of the Gradient Boosting Classifer and 
# the Random Forest Gradient both provided in scikit-learn

# The input: 
#    pandas.dataframe with columns: This should be in the same format as downloaded from the website

# The data frame go through the following pipeliine:
# 1. Cleaning
# 2. Creation of derived variables
# 3. Making predictions

# The output:
#    pandas.Series, two files are saved on disk,  submission.csv and cleaned_data.csv respectively.

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os, json, requests, pickle
from scipy.stats import skew
from shapely.geometry import Point,Polygon,MultiPoint,MultiPolygon
from scipy.stats import ttest_ind, f_oneway, lognorm, levy, skew, chisquare
from sklearn.preprocessing import normalize, scale
from sklearn import metrics
from tabulate import tabulate #pretty print of tables. source: http://txt.arboreus.com/2013/03/13/pretty-print-tables-in-python.html
from shapely.geometry import Point,Polygon,MultiPoint


import warnings
warnings.filterwarnings('ignore')

def read_me():
    """
    This is a function to print a read me instruction
    """
    print ("=========Introduction=========\n\nUse this code to predict the percentage tip expected after a trip in NYC green taxi. \nThe code is a predictive model that was built and trained on top of the Gradient Boosting Classifer and the Random Forest Gradient both provided in scikit-learn\n\nThe input: \npandas.dataframe with columns:This should be in the same format as downloaded from the website\n\nThe data frame go through the following pipeline:\n\t1. Cleaning\n\t2. Creation of derived variables\n\t3. Making predictions\n\nThe output:\n\tpandas.Series, two files are saved on disk,  submission.csv and cleaned_data.csv respectively.\n\nTo make predictions, run 'tip_predictor.make_predictions(data)', where data is any 2015 raw dataframe fresh from http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml\nRun tip_predictor.read_me() for further instructions\n")


# define a function to clean a loaded dataset
def __clean_data__(adata):
    """
    This function cleans the input dataframe adata:
    . drop Ehail_fee [99% transactions are NaNs]
    . impute missing values in Trip_type
    . replace invalid data by most frequent value for RateCodeID and Extra
    . encode categorical to numeric
    . rename pickup and dropff time variables (for later use)
    
    input:
        adata: pandas.dataframe
    output: 
        pandas.dataframe

    """
    ## make a copy of the input
    data = adata.copy()
    ## drop Ehail_fee: 99% of its values are NaNs
    if 'Ehail_fee' in data.columns:
        data.drop('Ehail_fee',axis=1,inplace=True)

    ##  replace missing values in Trip_type with the most frequent value 1
    data['Trip_type '] = data['Trip_type '].replace(np.NaN,1)
    
    ## replace all values that are not allowed as per the variable dictionary with the most frequent allowable value
    # remove negative values from Total amound and Fare_amount
    data.Total_amount = data.Total_amount.abs()
    data.Fare_amount = data.Fare_amount.abs()
    data.improvement_surcharge = data.improvement_surcharge.abs()
    data.Tip_amount = data.Tip_amount.abs()
    data.Tolls_amount = data.Tolls_amount.abs()
    data.MTA_tax = data.MTA_tax.abs()
    
    # RateCodeID
    indices_oi = data[~((data.RateCodeID>=1) & (data.RateCodeID<=6))].index
    data.loc[indices_oi, 'RateCodeID'] = 2 # 2 = Cash payment was identified as the common method
    
    # Extra
    indices_oi = data[~((data.Extra==0) | (data.Extra==0.5) | (data.Extra==1))].index
    data.loc[indices_oi, 'Extra'] = 0 # 0 was identified as the most frequent value
    
    # Total_amount: the minimum charge is 2.5, so I will replace every thing less than 2.5 by the median 11.76 (pre-obtained in analysis)
    indices_oi = data[(data.Total_amount<2.5)].index
    data.loc[indices_oi,'Total_amount'] = 11.76
    
    # encode categorical to numeric (I avoid to use dummy to keep dataset small)
    if data.Store_and_fwd_flag.dtype.name != 'int64':
        data['Store_and_fwd_flag'] = (data.Store_and_fwd_flag=='Y')*1
    
    # rename time stamp variables and convert them to the right format
    data.rename(columns={'lpep_pickup_datetime':'Pickup_dt','Lpep_dropoff_datetime':'Dropoff_dt'},inplace=True)
    data['Pickup_dt'] = data.Pickup_dt.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    data['Dropoff_dt'] = data.Dropoff_dt.apply(lambda x:dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    return data


# Function to run the feature engineering
def __engineer_features__(adata):
    """
    This function create new variables based on present variables in the dataset adata. It creates:
    . Week: int {1,2,3,4,5}, Week a transaction was done
    . Week_day: int [0-6], day of the week a transaction was done
    . Month_day: int [0-30], day of the month a transaction was done
    . Hour: int [0-23], hour the day a transaction was done
    . Shift type: int {1=(7am to 3pm), 2=(3pm to 11pm) and 3=(11pm to 7am)}, shift of the day  
    . Speed_mph: float, speed of the trip
    . Tip_percentage: float, target variable
    . With_tip: int {0,1}, 1 = transaction with tip, 0 transction without tip
    
    input:
        adata: pandas.dataframe
    output: 
        pandas.dataframe
    """
    
    # make copy of the original dataset
    data = adata.copy()
    
    # derive time variables
    ref_week = dt.datetime(2015,9,1).isocalendar()[1] # first week of september in 2015
    data['Week'] = data.Pickup_dt.apply(lambda x:x.isocalendar()[1])-ref_week+1
    data['Week_day']  = data.Pickup_dt.apply(lambda x:x.isocalendar()[2])
    data['Month_day'] = data.Pickup_dt.apply(lambda x:x.day)
    data['Hour'] = data.Pickup_dt.apply(lambda x:x.hour)
    #data.rename(columns={'Pickup_hour':'Hour'},inplace=True)

    # create shift variable:  1=(7am to 3pm), 2=(3pm to 11pm) and 3=(11pm to 7am)
    data['Shift_type'] = np.NAN
    data.loc[data[(data.Hour>=7) & (data.Hour<15)].index,'Shift_type'] = 1
    data.loc[data[(data.Hour>=15) & (data.Hour<23)].index,'Shift_type'] = 2
    data.loc[data[data.Shift_type.isnull()].index,'Shift_type'] = 3
    
    # Trip duration 
    data['Trip_duration'] = ((data.Dropoff_dt-data.Pickup_dt).apply(lambda x:x.total_seconds()/60.))
    
    # create direction variable Direction_NS. 
    # This is 2 if taxi moving from north to south, 1 in the opposite direction and 0 otherwise
    data['Direction_NS'] = (data.Pickup_latitude>data.Dropoff_latitude)*1+1
    indices = data[(data.Pickup_latitude == data.Dropoff_latitude) & (data.Pickup_latitude!=0)].index
    data.loc[indices,'Direction_NS'] = 0

    # create direction variable Direction_EW. 
    # This is 2 if taxi moving from east to west, 1 in the opposite direction and 0 otherwise
    data['Direction_EW'] = (data.Pickup_longitude>data.Dropoff_longitude)*1+1
    indices = data[(data.Pickup_longitude == data.Dropoff_longitude) & (data.Pickup_longitude!=0)].index
    data.loc[indices,'Direction_EW'] = 0
    
    # create variable for Speed
    data['Speed_mph'] = data.Trip_distance/(data.Trip_duration/60)
    # replace all NaNs values and values >240mph by a values sampled from a random distribution of 
    # mean 12.9 and  standard deviation 6.8mph. These values were extracted from the distribution
    indices_oi = data[(data.Speed_mph.isnull()) | (data.Speed_mph>240)].index
    data.loc[indices_oi,'Speed_mph'] = np.abs(np.random.normal(loc=12.9,scale=6.8,size=len(indices_oi)))
    
    # Create a new variable to check if a trip originated in Upper Manhattan
    data['U_manhattan'] = data[['Pickup_latitude','Pickup_longitude']].apply(lambda r:is_within_bbox((r[0],r[1])),axis=1)
    
    # create tip percentage variable
    data['Tip_percentage'] = 100*data.Tip_amount/data.Total_amount
    
    # create with_tip variable
    data['With_tip'] = (data.Tip_percentage>0)*1

    return data


# collected bounding box points
umanhattan = [(40.796937, -73.949503),(40.787945, -73.955822),(40.782772, -73.943575),
              (40.794715, -73.929801),(40.811261, -73.934153),(40.835371, -73.934515),
              (40.868910, -73.911145),(40.872719, -73.910765),(40.878252, -73.926350),
              (40.850557, -73.947262),(40.836225, -73.949899),(40.806050, -73.971255)]

poi = Polygon(umanhattan)
# create a function to check if a location is located inside Upper Manhattan
def is_within_bbox(loc,poi=poi):
    """
    This function checks if a location loc with lat and lon is located within the polygon of interest
    input:
    loc: tuple, (latitude, longitude)
    poi: shapely.geometry.Polygon, polygon of interest
    """
    return 1*(Point(loc).within(poi))


def __predict_tip__(transaction):
    """
    This function predicts the percentage tip expected on 1 transaction
    transaction: pandas.dataframe
    instead of calling this function immediately, consider calling it from "make_predictions"
    """
    # load models
    with open('my_classifier.pkl','rb') as fid:
        classifier = pickle.load(fid)
        fid.close()
    with open('my_regressor.pkl','rb') as fid:
        regressor = pickle.load(fid)
        fid.close()
        
    cls_predictors = ['Payment_type','Total_amount','Trip_duration','Speed_mph','MTA_tax',
                      'Extra','Hour','Direction_NS', 'Direction_EW','U_manhattan']
    reg_predictors = ['Total_amount', 'Trip_duration', 'Speed_mph']
    
    # classify transactions
    clas = classifier.predict(transaction[cls_predictors])
    
    # estimate and return tip percentage
    return clas*regressor.predict(transaction[reg_predictors])

def evaluate_predictions():
    """
    This looks for cleaned and predicted data set on disk and compare them
    """
    cleaned = pd.read_csv('cleaned_data.csv')
    predictions = pd.read_csv('submission.csv')
    print "mean squared error:", metrics.mean_squared_error(cleaned.Tip_percentage,predictions.predictions)
    print "r2 score:", metrics.r2_score(cleaned.Tip_percentage,predictions.predictions)

def make_predictions(data):
    """
    This makes sure that data has the right format and then send it to the prediction model to be predicted
    data: pandas.dataframe, raw data from the website
    the outputs are saved on disk: submissions and cleaned data saved as submission.csv and cleaned_data.csv respectively 
    """
    print "cleaning ..."
    data = __clean_data__(data)
    print "creating features ..."
    data = __engineer_features__(data)
    print "predicting ..."
    preds = pd.DataFrame(__predict_tip__(data),columns=['predictions'])
    preds.index = data.index
    pd.DataFrame(data.Tip_percentage,columns=['Tip_percentage']).to_csv('cleaned_data.csv',index=True)
    preds.to_csv('submission.csv',index=True)
    print "submissions and cleaned data saved as submission.csv and cleaned_data.csv respectively"
    print "run evaluate_predictions() to compare them"
    


read_me()
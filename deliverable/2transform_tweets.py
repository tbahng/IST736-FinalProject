"""
This file creates a single data frame from tweet text 
from "data/extract_candidate_tweets.json" and 
"data/extract_noncandidate_tweets.json"
so that data can be explored, analyzed and modeled.

Relevant tweet text for modeling is as follows:
    Tweets that were created from 2018-01-01 to 2018-12-31
However, due to the small sample size of that data,
candidacy announcement dates will be added as an attribute to 
help identify any tweets that were created before this date

Key Fields in tweets dataframe:
- handle: screen name of user
- user_name: name of user
- candidate_2020
- join_2020
- candidate_2016
- join_2016
- tweet_text
- tweet_created_at

Results saved to 'data/data.csv'
"""

import bz2 
import json 
import pandas as pd 
import datetime

def get_date(created_at):
    """Function to convert Twitter created_at to date format
    Argument:
        created_at {[str]} -- [raw tweet creation date time stamp]
    Returns:
        [str] -- [date e.g. '2020-04-18']
    """
    dt_obj = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
    #dt_str = datetime.datetime.strftime(dt_obj, '%Y-%m-%d')
    return dt_obj

def check_date(created_at, start, end):
    """Function to check whether twitter created_at time stamp is between two dates
    Argument:
        created_at {[str]} -- [raw tweet creation date time stamp]
        start {[str]} -- [start date of timeframe e.g. '2018-01-01']
        end {[str]} -- [end date of timeframe e.g. '2019-03-31']
    Returns:
        [bool] -- [True or False]
    """
    x = get_date(created_at)
    return x <= end and x >= start

##################################################
# read People.xlsx for the original data
##################################################
fname = "data/People.xlsx"
df = pd.read_excel(fname, sheet_name = 'All')

##################################################
# read candidate tweets from json
##################################################
fname = "data/extract_candidate_tweets.json"
with bz2.BZ2File(fname, 'r') as fin:
    candidate_data = json.loads(fin.read().decode('utf-8'))
print("Candidate tweets collected: {:d}".format(len(candidate_data.keys())))

##################################################
# read noncandidate tweets from json
##################################################
fname = "data/extract_noncandidate_tweets.json"
with bz2.BZ2File(fname, 'r') as fin:
    noncandidate_data = json.loads(fin.read().decode('utf-8'))
print("NonCandidate tweets collected: {:d}".format(len(noncandidate_data.keys())))
# exclude non-candidate keys with no data
keys_drop = []
for key in noncandidate_data.keys():
    if len(noncandidate_data[key]) == 0:
        keys_drop.append(key)
for key in keys_drop:
    print("Dropping key for {:s} due to missing data".format(key))
    del noncandidate_data[key]

##################################################
# combine candidate_data and noncandidate_data
##################################################
all_tweets_collected = {**candidate_data, **noncandidate_data}

##################################################
# convert tweets to dataframe
##################################################
def to_df(handle):
    criteria = df['Handle'] == handle
    tmp = df.loc[criteria,:]
    dat = pd.DataFrame(all_tweets_collected[handle])
    dat['created_at'] = dat['created_at'].apply(get_date)
    dat['handle'] = handle 
    dat['user_name'] = tmp['Name'].values[0]
    dat['candidate_2020'] = tmp['Ran 2020'].isin([1]).values[0]
    dat['join_2020'] = tmp['2020 Join'].values[0]
    dat['candidate_2016'] = tmp['Ran 2016'].isin([1]).values[0]
    dat['join_2016'] = tmp['2016 Join'].values[0]
    return dat.iloc[:, [2,3,4,5,6,7,0,1]]

df_list = [to_df(handle) for handle in all_tweets_collected.keys()]
df_tweets = pd.concat(df_list)

##################################################
# save to file
##################################################
fname = 'data/data.csv'
df_tweets.to_csv(fname, index = None)


##################################################
# sample data for deliverable
##################################################
fname = 'data/sample_data.csv'
df_tweets.head(100).to_csv(fname, index = None)
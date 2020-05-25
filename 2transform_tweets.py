"""
This file adds a column 'text' to 'data/People.xlsx'
for tweet text from "data/extract_candidate_tweets.json" and 
"data/extract_noncandidate_tweets.json"
so that data can be explored, analyzed and modeled.

Addresable tweet text is as follows:
    Tweets that were created from 2018-01-01 to 2018-12-31

Records of 'data/People.xlsx' is subset to only handles for which
there is tweet text.

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
    dt_str = datetime.datetime.strftime(dt_obj, '%Y-%m-%d')
    return dt_str

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
# create new dictionary of only tweets 
# from Jan-Dec 2018
##################################################
tweets = {}
for handle in all_tweets_collected.keys():
    tweets_list = [] # initialize tweets to keep
    # only keep tweets within specified timeframe
    for tweet in all_tweets_collected[handle]:
        if check_date(tweet['created_at'], '2018-01-01', '2018-12-31'):
            tweets_list.append(tweet['text'])
        else:
            continue 
    if len(tweets_list) > 0:
        tweets[handle] = tweets_list

# list of all twitter handles with tweets
handle_list = list(tweets.keys())

# subset people dataframe for only these handles
dfsub = df.loc[df['Handle'].isin(handle_list),:]

# add column for tweet text to people dataframe
dfsub['text'] = [' '.join(tweets[handle]) for handle in dfsub['Handle']]

##################################################
# save to file
##################################################
fname = 'data/data.csv'
dfsub.to_csv(fname, index = None)
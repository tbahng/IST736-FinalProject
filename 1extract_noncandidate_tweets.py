"""This file reads People.xlsx and queries Twitter for every handle/screen name in the file.
"""

import tweepy
import os
import sys
import bz2 
import json 
import pandas as pd 
import numpy as np 
import re 
import datetime
import random

"""Authorization codes and data tools for using the Twitter REST API
These are authorization codes from personal Twitter developer account
https://apps.twitter.com/
"""
CONSUMER_KEY = ''
CONSUMER_SECRET = ''
OAUTH_TOKEN = ''
OAUTH_SECRET = ''

def oauth_login():
    """login to Twitter with ordinary rate limiting
    needs defined authorization codes for personal twitter developer application
    CONSUMER_KEY (consumer api key)
    CONSUMER_SECRET (consumer api secret key)
    OAUTH_TOKEN (access token)
    OAUTH_SECRET (access token secret)
    Returns:
        [tweepy.api.API] -- [tweepy api]
    """
    # get the authorization from Twitter and save in the Tweepy package
    auth = tweepy.OAuthHandler(CONSUMER_KEY,CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN,OAUTH_SECRET)
    tweepy_api = tweepy.API(auth)
    # if a null api is returned, give error message
    if (not tweepy_api):
        print ("Problem Connecting to API with OAuth")
        # return the Twitter api object that allows access for the Tweepy api functions
    return tweepy_api

# setup API
api = oauth_login()

###########################################################
# read People.xlsx to acquire twitter handle information
# extract english language tweet text and created_at data
###########################################################
fname = "data/People.xlsx"
if os.path.isfile(fname):    
    people = pd.read_excel(fname, sheet_name="All")
    people['Handle'] = people['Handle'].fillna('-')       
    # handles for 2020 candidates
    hlist_2020_candidates = sorted([h for h in people.loc[people['Ran 2020'] == 1, 'Handle'] if not h == '-'])
    # handles for 2016 candidates
    hlist_2016_candidates = sorted([h for h in people.loc[people['Ran 2016'] == 1, 'Handle'] if not h == '-'])
    # handles for all candidates
    hlist_all_candidates = sorted(list(set(hlist_2016_candidates + hlist_2020_candidates)))
    # handles for non candidates in both 2016 and 2020
    hlist_non_candidates = sorted([h for h in people.loc[(people['Ran 2020'] == 0) & (people['Ran 2016'] == 0), 'Handle'] if not h == '-'])
    # balanced random sample of non candidates
    random.seed(11)
    random.shuffle(hlist_non_candidates)
    # number of candidate handles
    limit = len(hlist_all_candidates)
    # list of all handles
    hlist = hlist_non_candidates     
    print("Getting tweets for {:d} Twitter handles".format(limit))    
    # dictionary containing list of tweets respective to handles (keys)    
    from time import time
    t0 = time() # start the cloc
    collection_counter = 0
    user_tweets = {}
    for handle in hlist:
        if collection_counter < limit:
            try:
                # get all available tweets from index for this twitter handle
                search_results = [status for status in tweepy.Cursor(api.user_timeline, id = handle, wait_on_rate_limit=True).items()]
                # format tweet as json
                tweets = [tweet._json for tweet in search_results]
                # filter for english tweets
                tweets = [tweet for tweet in tweets if tweet['lang'] == 'en']
                # subset to just text and created_at
                tweets = [{k: tweet[k] for k in ('text','created_at')} for tweet in tweets]
                # add tweet to user_tweets dictionary
                user_tweets[handle] = tweets
                # increment collection counter by 1
                collection_counter += 1                
            except Exception as e:
                print(e, "handle: {:s}".format(handle))
            # continue looping
            continue
        else:
            break
    print("done in %0.3fs" % (time() - t0)) # time elapsed
    print("number of users collected: {:d}".format(len(user_tweets.keys())))
else:
    sys.exit("File not found: People.xlsx")

###########################################################
# validate tweet info
###########################################################
# summarize total number of tweets collected
tweets_count = 0
for user in user_tweets.keys():
    for tweet in user_tweets[user]:
        tweets_count += 1
print("A total of {:d} tweets have been collected".format(tweets_count))

"""Functions for exploring tweets
get_date
check_date
"""
#import time
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

# print the number of tweets and timeframe of tweets by user
for handle in user_tweets.keys():
    try:
        tweets = user_tweets[handle]
        datelist = [get_date(tweet['created_at']) for tweet in tweets]
        start = min(datelist)
        end = max(datelist)
        print(handle, len(user_tweets[handle]), "from: {:s} to: {:s}".format(start, end))
    except Exception as e:
        print(e, "handle: {:s}".format(handle))

###########################################################
# save tweets to json file
###########################################################
fname = "data/extract_noncandidate_tweets.json"

with bz2.BZ2File(fname, 'w') as fout:
    fout.write(json.dumps(user_tweets).encode('utf-8'))
print("Results saved in {:s}".format(fname))

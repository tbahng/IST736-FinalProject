import tweepy 
import os
import bz2
import json
import time
from datetime import datetime, date, time, timedelta
from collections import Counter
import numpy as np
import pandas as pd
import re

# authorization codes here from personal Twitter developer account
# https://apps.twitter.com/
CONSUMER_KEY = '3OBvg0FfoJcE2vUkho22TRlJT'
CONSUMER_SECRET = 'UIcPiD5YmyVYgW1VaAYtq2QndvA0oVUQpEBv6dhXEDitBDlN5m'
OAUTH_TOKEN = '1196644023189958656-h7gx9NtxoPAs1GnALvUnU3GivRaWE5'
OAUTH_SECRET = 'AvkOHyCGahfX9UGCUEJEDNyjgwK11EX5tzZ1kISkNV4ub'

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

def get_tweets(api, who, max_results=20):
    """Uses the tweepy Cursor to wrap a twitter api search for the query string
    returns json formatted results
    Arguments:
        api {[tweepy.api.API]} -- [tweepy api]
        query {[str]} -- [twitter handle]
    
    Keyword Arguments:
        start {str} -- [timeframe start e.g. '20200411']
        end {str} -- [timeframe end e.g. '20200412']
        max_results {int} -- [number of tweets] (default: {20})
    
    Returns:
        [list] -- [list of dictionaries where each element is a tweet]
    """
    start = datetime.strptime(start, '%Y%m%d')
    end = datetime.strptime(end, '%Y%m%d')
    # the first search initializes a cursor, stored in the metadata results,
    # that allows next searches to return additional tweets
    search_results = [status for status in tweepy.Cursor(api.user_timeline, id=query).items(max_results) if status.created_at < end and status.created_at > start]  
    # for each tweet, get the json representation
    tweets = [tweet._json for tweet in search_results]  
    return tweets

# Data source includes index, people names, class labels, and twitter handle
People = pd.read_excel('data/People.xlsx', sheet_name='All') # data source

# handle metadata
Handle = pd.read_csv('data/handle_data_demo.csv')

#https://bhaskarvk.github.io/2015/01/how-to-use-twitters-search-rest-api-most-effectively./
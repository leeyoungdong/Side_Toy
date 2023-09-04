import sqlite3
import pandas as pd
import tweepy
import os

DB_FILENAME = 'project.db'
DB_FILEPATH = os.path.join(os.getcwd(), DB_FILENAME)

conn = sqlite3.connect(DB_FILENAME)
cur = conn.cursor()


def connect_api():

    consumer_key = 'xZnqscfausHjRdvRoogHmuX9X'
    consumer_secret = 'kMHN9D8tPi9xv6wtizkbaQqaDNFUIM0k0ZFJ29qMklDj4iNAJf'
    access_token = '1556861388739342336-ZpocS7S0lRPY0mh8Eilma2tDWdtCBR'
    access_token_secret = 'FUlhGceK8H7QMSAv3jrAV7KLBuyEHUEyWL6v42AdITFwR'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    return api


def get_tweets(api, username):

    tweets = None    
    tweets = api.user_timeline(username, tweet_mode='extended')

    return tweets

# list 를 append.를 해줘야함.
# execute전에 string 형식으로 바꿔줘야지 들어가쥐

def first_data(name):
    tweet_list = []
    api = connect_api()

    cur.execute("DROP TABLE IF EXISTS twitters")

    for status in tweepy.Cursor(api.user_timeline, screen_name = name).items(10000):
        temp_list = [status.text, status.retweet_count, status.favorite_count]
        tweet_list.append(temp_list)
    df = pd.DataFrame(tweet_list, columns=['Tweets', 'Retweets', 'Likes'])

    print(len(df))
    df.to_sql('twitters', conn)
    conn.commit()

    return df


def data(name):

    tweet_list = []
    api = connect_api()

    for status in tweepy.Cursor(api.user_timeline, screen_name = name).items(10000):
        temp_list = [status.text, status.retweet_count, status.favorite_count]
        tweet_list.append(temp_list)

    df = pd.DataFrame(tweet_list, columns=['Tweets', 'Retweets', 'Likes'])
    print(len(df))
    df.to_sql('twitters', conn, if_exists = 'append')
    conn.commit()

    return df


# list 를 append.를 해줘야함.
#  string 형식으로 바꿔줘야지 들어가쥐


# first_data('YouTube')
# data('DisneyPlus')
# data('Sony')

# data('Marvel')
# data('WBHomeEnt')
# data('DCComics')

# data('UniversalPics')
# data('20thHomeEnt')
# data('ParamountPics')

# data('Dreamworks')
# data('DisneyStudios')




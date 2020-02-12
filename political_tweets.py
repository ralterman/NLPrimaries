# !pip install -U -q PyDrive
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
# ​
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# ​
# from google.colab import drive
# drive.mount('/content/gdrive')
#
# import twint
# import nest_asyncio
# nest_asyncio.apply()
#
#
# c = twint.Config()
# c.Search = "Buttigieg OR Mayor Pete OR @PeteButtigieg AND -Biden AND -Warren AND -Klobuchar AND -Bloomberg AND -Sanders AND -Bernie AND -@ewarren"\
 # "AND -@SenWarren AND -@MikeBloomberg AND -@JoeBiden AND -@amyklobuchar AND -@SenAmyKlobuchar AND -@BernieSanders AND -@SenSanders"
# c.Store_csv = True
# c.Verified = True
# c.Output = 'buttigieg.csv'
# c.Since = '2020-01-01'


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import cm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import RegexpTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation

pd.set_option('display.max_columns', None)


def drop_col(df):
    new_df = df.drop(['id', 'conversation_id', 'created_at','timezone','user_id','place','urls','photos','replies_count','retweets_count', 'likes_count','cashtags',
                      'retweet','quote_url','video','near','geo','source','user_rt_id','user_rt','retweet_id','retweet_date','translate','trans_src','trans_dest'], axis=1)
    return new_df


buttigieg = pd.read_csv('buttigieg.csv')
buttigieg = drop_col(buttigieg)

warren = pd.read_csv('warren.csv')
warren = drop_col(warren)

bernie = pd.read_csv('bernie.csv')
bernie = drop_col(bernie)

biden = pd.read_csv('biden.csv')
biden = drop_col(biden)

amy = pd.read_csv('klobuchar.csv')
amy = drop_col(amy)

bloomberg = pd.read_csv('bloomberg.csv')
bloomberg = drop_col(bloomberg)

trump = pd.read_csv('trump.csv')
trump = drop_col(trump)


candidates = [bernie, buttigieg, amy, warren, biden, bloomberg, trump]

#-----------------------------------------------------------------------------------------------


# TOKENIZE EVERY TWEET
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')

def tokenize(tweets):
    tokenized_tweets = []
    for i in tweets:
        token_tweet = tokenizer.tokenize(i)
        tokenized_tweets.append(token_tweet)
    return tokenized_tweets

token_amy = tokenize(amy['tweet'])
token_bloomberg = tokenize(bloomberg['tweet'])
token_buttigieg = tokenize(buttigieg['tweet'])
token_warren = tokenize(warren['tweet'])
token_biden = tokenize(biden['tweet'])
token_bernie = tokenize(bernie['tweet'])

# CREATE WORD BANK FOR EACH CANDIDATE
bloomberg_bank = word_bank(token_bloomberg)
buttigieg_bank = word_bank(token_buttigieg)
warren_bank = word_bank(token_warren)
biden_bank = word_bank(token_biden)
bernie_bank = word_bank(token_bernie)

# Remove links
def remove_urls(vTEXT):
    vTEXT = re.sub(r'https?:\/\/.*\s?', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

def remove_twitpics(vTEXT):
    vTEXT1 = re.sub(r'pic\.twitter\.com\/.*\s?', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT1)

def remove_links(tweet):
    no_urls = []
    no_twitpics = []
    for i in tweet:
        removed_urls = remove_urls(i)
        no_urls.append(removed_urls)
        for j in no_urls:
            removed_twitpics = remove_twitpics(j)
            no_twitpics.append(removed_twitpics)
    return no_twitpics


#-----------------------------------------------------------------------------------------------


for democrat in candidates:
    democrat.tweet = democrat.tweet.str.replace(r'https?:\/\/.*\s?', '')
    democrat.tweet = democrat.tweet.str.replace(r'pic\.twitter\.com\/.*\s?', '')


analyzer = SentimentIntensityAnalyzer()


def positive_sentiment(row):
    scores = analyzer.polarity_scores(row['tweet'])
    positive_sentiment = scores['pos']
    return positive_sentiment

def neutral_sentiment(row):
    scores = analyzer.polarity_scores(row['tweet'])
    neutral_sentiment = scores['neu']
    return neutral_sentiment

def negative_sentiment(row):
    scores = analyzer.polarity_scores(row['tweet'])
    negative_sentiment = scores['neg']
    return negative_sentiment

def compound_sentiment(row):
    scores = analyzer.polarity_scores(row['tweet'])
    compound_sentiment = scores['compound']
    return compound_sentiment


for democrat in candidates:
    democrat['positive_sentiment'] = democrat.apply(positive_sentiment, axis=1)
    democrat['neutral_sentiment'] = democrat.apply(neutral_sentiment, axis=1)
    democrat['negative_sentiment'] = democrat.apply(negative_sentiment, axis=1)
    democrat['compound_sentiment'] = democrat.apply(compound_sentiment, axis=1)



names = ['Sanders', 'Buttigieg', 'Klobuchar', 'Warren', 'Biden', 'Bloomberg', 'Trump']

idx = 0
sentiment = {}
for democrat in candidates:
    sentiment[names[idx]] = {'positive': democrat['positive_sentiment'].mean(), 'neutral': democrat['neutral_sentiment'].mean(),
                              'negative': democrat['negative_sentiment'].mean(), 'compound': democrat['compound_sentiment'].mean()}
    idx += 1

sentiment


def positive(row):
    if row['positive_sentiment'] > row['negative_sentiment']:
        return 1
    else:
        return 0

def negative(row):
    if row['negative_sentiment'] > row['positive_sentiment']:
        return 1
    else:
        return 0


for democrat in candidates:
    democrat['positive_tweet'] = democrat.apply(positive, axis=1)
    democrat['negative_tweet'] = democrat.apply(negative, axis=1)



count = 0
pos_neg = {}
for democrat in candidates:
    pos_neg[names[count]] = {'positive': (democrat['positive_tweet'].sum() / len(democrat)), 'negative': (democrat['negative_tweet'].sum() / len(democrat))}
    count += 1

pos_neg


sentiment_df = pd.DataFrame(pos_neg).T
sentiment_df


sentiment_df.plot(kind='bar', figsize=(15,8), color=['b', 'r'])
plt.title('Sentiment Analysis of Tweets Mentioning Candidates', fontsize=20)
plt.legend(fontsize=15)
plt.xticks(fontsize=13, rotation=0)
plt.yticks(fontsize=13)
plt.xlabel('Candidate', fontsize=18)
plt.ylabel('% of Tweets', fontsize=18)


#-----------------------------------------------------------------------------------------------

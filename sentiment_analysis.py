import pandas as pd
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import LatentDirichletAllocation
from langdetect import detect
from langdetect import detect_langs

pd.set_option('display.max_columns', None)



# Drop unnecessary columns
def drop_col(df):
    new_df = df.drop(['id', 'conversation_id', 'created_at','timezone','user_id','place','urls','photos','replies_count','retweets_count', 'likes_count','cashtags',
                      'retweet','quote_url','video','near','geo','source','user_rt_id','user_rt','retweet_id','retweet_date','translate','trans_src','trans_dest'], axis=1)
    return new_df


# Read in csv of tweets for all candidates (from twint_scraper.ipynb script) and drop aforementioned columns
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

# Remove links and pictures from tweets
for democrat in candidates:
    democrat.tweet = democrat.tweet.str.replace(r'https?:\/\/.*\s?', '')
    democrat.tweet = democrat.tweet.str.replace(r'pic\.twitter\.com\/.*\s?', '')


# Remove non-english tweets
def language(row):
    try:
        return detect(row['tweet'])
    except:
        return 0

for democrat in candidates:
    democrat['language'] = democrat.apply(language, axis=1)


buttigieg = (buttigieg[buttigieg['language'] == 'en']).reset_index(drop=True)
warren = warren[warren['language'] == 'en'].reset_index(drop=True)
bernie = bernie[bernie['language'] == 'en'].reset_index(drop=True)
biden = biden[biden['language'] == 'en'].reset_index(drop=True)
amy = amy[amy['language'] == 'en'].reset_index(drop=True)
bloomberg = bloomberg[bloomberg['language'] == 'en'].reset_index(drop=True)
trump = trump[trump['language'] == 'en'].reset_index(drop=True)


#-----------------------------------------------------------------------------------------------


# Create sentiment analysis functions
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


candidates = [bernie, buttigieg, amy, warren, biden, bloomberg, trump]

# Run sentiment analysis on every tweet for every candidate and add scores to the dataframe
for democrat in candidates:
    democrat['positive_sentiment'] = democrat.apply(positive_sentiment, axis=1)
    democrat['neutral_sentiment'] = democrat.apply(neutral_sentiment, axis=1)
    democrat['negative_sentiment'] = democrat.apply(negative_sentiment, axis=1)
    democrat['compound_sentiment'] = democrat.apply(compound_sentiment, axis=1)



names = ['Sanders', 'Buttigieg', 'Klobuchar', 'Warren', 'Biden', 'Bloomberg', 'Trump']

# Create sentiment analysis dictionary with averages
idx = 0
sentiment = {}
for democrat in candidates:
    sentiment[names[idx]] = {'positive': democrat['positive_sentiment'].mean(), 'neutral': democrat['neutral_sentiment'].mean(),
                              'negative': democrat['negative_sentiment'].mean(), 'compound': democrat['compound_sentiment'].mean()}
    idx += 1

sentiment


 # Mark each tweet as either positive or negative depending on higher sentiment value
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


# Create dictionary with average tweet positivity and negativity per candidate
count = 0
pos_neg = {}
for democrat in candidates:
    pos_neg[names[count]] = {'positive': (democrat['positive_tweet'].sum() / len(democrat)), 'negative': (democrat['negative_tweet'].sum() / len(democrat))}
    count += 1

pos_neg


sentiment_df = pd.DataFrame(pos_neg).T
sentiment_df


# Plot these positive and negative scores in double bar chart
sentiment_df.plot(kind='bar', figsize=(15,8), color=['b', 'r'])
plt.title('Sentiment Analysis of Tweets Mentioning Candidates', fontsize=20)
plt.legend(fontsize=15)
plt.xticks(fontsize=13, rotation=0)
plt.yticks(fontsize=13)
plt.xlabel('Candidate', fontsize=18)
plt.ylabel('% of Tweets', fontsize=18)


#-----------------------------------------------------------------------------------------------

# Export cleaned csv files for topic modeling
buttigieg.to_csv('buttigieg_cleaned.csv', index=False)
warren.to_csv('warren_cleaned.csv', index=False)
bernie.to_csv('bernie_cleaned.csv', index=False)
biden.to_csv('biden_cleaned.csv', index=False)
amy.to_csv('klobuchar_cleaned.csv', index=False)
bloomberg.to_csv('bloomberg_cleaned.csv', index=False)
trump.to_csv('trump_cleaned.csv', index=False)

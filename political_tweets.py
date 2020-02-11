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
# c.Search = "Buttigieg OR Mayor Pete OR @PeteButtigieg AND -Biden AND -Warren AND -Klobuchar AND -Bloomberg AND -Yang AND -Sanders AND -Bernie AND -@ewarren"\
 # "AND -@SenWarren AND -@MikeBloomberg AND -@JoeBiden AND -@amyklobuchar AND -@SenAmyKlobuchar AND -@BernieSanders AND -@SenSanders AND -@AndrewYang"
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

pd.set_option('display.max_columns', None)


def drop_col(df):
    new_df = df.drop(['id', 'conversation_id', 'created_at','timezone','user_id','place','urls','photos','replies_count','retweets_count', 'likes_count','cashtags',
                      'retweet','quote_url','video','near','geo','source','user_rt_id','user_rt','retweet_id','retweet_date','translate','trans_src','trans_dest'], axis=1)
    return new_df


buttigieg = pd.read_csv('buttigieg.csv')
buttigieg = drop_col(buttigieg)

warren = pd.read_csv('warren.csv')
warren = drop_col(warren)

yang = pd.read_csv('yang.csv')
yang = drop_col(yang)

bernie = pd.read_csv('bernie.csv')
bernie = drop_col(bernie)

biden = pd.read_csv('biden.csv')
biden = drop_col(biden)

amy = pd.read_csv('klobuchar.csv')
amy = drop_col(amy)

bloomberg = pd.read_csv('bloomberg.csv')
bloomberg = drop_col(bloomberg)

candidates = [buttigieg, warren, yang, bernie, biden, amy, bloomberg]


#-----------------------------------------------------------------------------------------------


# tokenize and remove stopwords
nltk.download('stopwords')
stop_words=set(stopwords.words("english"))
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')

yang_token_tweet = []
for i in yang['tweet']:
    token_tweet = tokenizer.tokenize(i)
    yang_token_tweet.append(token_tweet)

yang['token_tweet'] = yang_token_tweet


#-----------------------------------------------------------------------------------------------


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


bernie.head()

names = ['Buttigieg', 'Warren', 'Yang', 'Bernie', 'Biden', 'Klobuchar', 'Bloomberg']
idx = 0
sentiment = {}
for democrat in candidates:
    sentiment[names[idx]] = {'positive': democrat['positive_sentiment'].mean(), 'neutral': democrat['neutral_sentiment'].mean(),
                              'negative': democrat['negative_sentiment'].mean(), 'compound': democrat['compound_sentiment'].mean()}
    idx += 1

sentiment

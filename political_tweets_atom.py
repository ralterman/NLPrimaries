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
# c.Search = "Buttigieg OR Mayor Pete OR @PeteButtigieg AND -Biden AND -Warren AND -Klobuchar AND -Bloomberg AND -Yang AND -Sanders AND -Bernie AND -@ewarren AND -@SenWarren AND -@MikeBloomberg AND -@JoeBiden AND -@amyklobuchar AND -@SenAmyKlobuchar AND -@BernieSanders AND -@SenSanders AND -@AndrewYang"
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


pd.set_option('display.max_columns', None)


def drop_col(df):
    new_df = df.drop(['id', 'conversation_id', 'created_at','timezone','user_id','place','urls','photos','replies_count','retweets_count','likes_count','cashtags','retweet','quote_url','video','near','geo','source','user_rt_id','user_rt','retweet_id','retweet_date','translate','trans_src','trans_dest'], axis=1)
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
candidates = [buttigieg, warren, bernie, biden, amy, bloomberg]

# Remove urls from each tweet
import re
def remove_urls (vTEXT):
    vTEXT = re.sub(r'https?:\/\/.*\s?', ' ', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

def remove_twitpics (vTEXT):
    vTEXT1 = re.sub(r'pic\.twitter\.com\/.*\s?', ' ', vTEXT, flags=re.MULTILINE)
    return(vTEXT1)

def remove_links(tweet):
    no_urls = []
    #no_twitpics = []
    for i in tweet:
        removed_urls = remove_urls(i)
        no_urls.append(removed_urls)
        #for j in no_urls:
        #    removed_twitpics = remove_twitpics(j)
        #    no_twitpics.append(removed_twitpics)
    #return no_twitpics

def no_links(row):
    return re.sub(r'https?:\/\/.*\s?', ' ', row['tweet'])

def no_pics(row):
    return re.sub(r'pic\.twitter\.com\/.*\s?', ' ', row['tweet'])

def no_numbers(row):
    return re.sub(r'\d.*\s?', '', row['tweet'])


for democrat in candidates:
    democrat['tweet'] = democrat.apply(no_links, axis=1)
    democrat['tweet'] = democrat.apply(no_pics, axis=1)
    democrat['tweet'] = democrat.apply(no_numbers, axis=1)


# TOKENIZE EVERY TWEET
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')

def tokenize(tweets):
    tokenized_tweets = []
    for i in tweets:
        token_tweet = tokenizer.tokenize(i)
        tokenized_tweets.append(token_tweet)
    return tokenized_tweets

def word_bank(token_tweets):
    all_words = []
    for i in token_tweets:
        for j in i:
            all_words.append(j)
    return all_words

token_amy = tokenize(amy['tweet'])
token_bloomberg = tokenize(bloomberg['tweet'])
token_buttigieg = tokenize(buttigieg['tweet'])
token_warren = tokenize(warren['tweet'])
token_biden = tokenize(biden['tweet'])
token_bernie = tokenize(bernie['tweet'])

amy_bank = word_bank(token_amy)
bloomberg_bank = word_bank(token_bloomberg)
buttigieg_bank = word_bank(token_buttigieg)
warren_bank = word_bank(token_warren)
biden_bank = word_bank(token_biden)
bernie_bank = word_bank(token_bernie)

# Lowercase all words
def lower(bank):
    lower_bank = []
    for i in bank:
        i = i.lower()
        lower_bank.append(i)
    bank = lower_bank
    return bank

amy_bank = lower(amy_bank)
bloomberg_bank = lower(bloomberg_bank)
buttigieg_bank = lower(buttigieg_bank)
warren_bank = lower(warren_bank)
biden_bank = lower(biden_bank)
bernie_bank = lower(bernie_bank)

# Delete stopwords
nltk.download('stopwords')
stop_words=set(stopwords.words("english"))
stop_words=(stopwords.words("english"))
political_stopwords = ['taupin','madoff', 'andrew', 'iowa', 'demdebate','mikebloomberg', 'ewarren', 'senwarren', 'berniesanders', 'sensanders', 'joebiden', 'petebuttigieg', 'sentinasmith', 'sen', 'yang', 'amyklobuchar', 'trump', 'elizabeth', 'warren', 'pete', 'peter', 'buttigieg', 'bloomberg', 'mike', 'michael', 'bernie', 'sanders', 'amy', 'klobuchar', 'joe','joseph', 'biden', 'i']

for i in range(len(political_stopwords)):
    stop_words.append(political_stopwords[i])


def stop(candidate):
    stpwrd = []
    for w in candidate:
        if w not in stop_words:
            stpwrd.append(w)
    return stpwrd

amy_bank = stop(amy_bank)
bloomberg_bank = stop(bloomberg_bank)
buttigieg_bank = stop(buttigieg_bank)
warren_bank = stop(warren_bank)
biden_bank = stop(biden_bank)
bernie_bank = stop(bernie_bank)

#LEMMATIZATION

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatized_tweet=[]
def lemmatize(bank):
    for w in bank:
        lemmatized_tweet.append(lemmatizer.lemmatize(w))
    return bank

amy_bank = lemmatize(amy_bank)
bloomberg_bank = lemmatize(bloomberg_bank)
buttigieg_bank = lemmatize(buttigieg_bank)

warren_bank = lemmatize(warren_bank)
biden_bank = lemmatize(biden_bank)
bernie_bank = lemmatize(bernie_bank)
banks = (amy_bank, bloomberg_bank, buttigieg_bank, warren_bank, biden_bank, bernie_bank)

#TF-IDF

# implementing it in python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# Convert a collection of text documents to a matrix of token counts

names = ['amy', 'bloomberg', 'buttigieg', 'warren', 'biden', 'bernie']
list_of_tuples = list(zip(names, banks))
df = pd.DataFrame(list_of_tuples, columns = ['names', 'wordbanks'])
df
list1 = df['wordbanks'][0]
str1 = ' '.join(list1)
list2 = df['wordbanks'][1]
str2 = ' '.join(list2)
list3 = df['wordbanks'][2]
str3 = ' '.join(list3)
list4 = df['wordbanks'][3]
str4 = ' '.join(list4)
list5 = df['wordbanks'][4]
str5 = ' '.join(list5)
list6 = df['wordbanks'][5]
str6 = ' '.join(list6)


string_list = [str1, str2, str3, str4, str5, str6]
vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(string_list)
cv = CountVectorizer()
word_count_vector=cv.fit_transform(string_list)
word_count_vector.shape
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)



# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])

# sort ascending
df_idf.sort_values(by=['idf_weights'])

# the lower the idf value, the more the words appear in every DataFrame.


count_vector=cv.transform(string_list)
tf_idf_vector=tfidf_transformer.transform(count_vector)
feature_names = cv.get_feature_names()

#get tfidf vector for first document
first_document_vector=tf_idf_vector[5]

#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
tfidf_df = df.sort_values(by=["tfidf"],ascending=False)
tfidf_df = tfidf_df[0:50]
# term frequency is weighted by IDF values by multiplying.
tfidf_df = tfidf_df.reset_index()
tfidf_df.columns = ['word', 'tfidf']

tfidf_df



from sklearn.feature_extraction.text import TfidfVectorizer

# settings that you use for count vectorizer will go here
tfidf_vectorizer=TfidfVectorizer(use_idf=True)

# just send in all your docs here
tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(string_list)


first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]

# place tf-idf values in a pandas data frame
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
sorted_df = df.sort_values(by=["tfidf"],ascending=False)

amy_vector = tfidf_vectorizer_vectors[0]
amy_df = pd.DataFrame(amy_vector.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
amy_sorted_df = amy_df.sort_values(by=["tfidf"],ascending=False)
bloomberg_vector = tfidf_vectorizer_vectors[1]
bloomberg_df = pd.DataFrame(bloomberg_vector.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
bloomberg_sorted_df = bloomberg_df.sort_values(by=["tfidf"],ascending=False)
buttigieg_vector = tfidf_vectorizer_vectors[2]
buttigieg_df = pd.DataFrame(buttigieg_vector.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
buttigieg_sorted_df = buttigieg_df.sort_values(by=["tfidf"],ascending=False)
warren_vector = tfidf_vectorizer_vectors[3]
warren_df = pd.DataFrame(warren_vector.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
warren_sorted_df = warren_df.sort_values(by=["tfidf"],ascending=False)
biden_vector = tfidf_vectorizer_vectors[4]
biden_df = pd.DataFrame(biden_vector.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
biden_sorted_df = biden_df.sort_values(by=["tfidf"],ascending=False)
bernie_vector = tfidf_vectorizer_vectors[5]
bernie_df = pd.DataFrame(bernie_vector.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
bernie_sorted_df = bernie_df.sort_values(by=["tfidf"],ascending=False)


df_list = [amy_common_df, bloomberg_common_df, buttigieg_common_df, warren_common_df, biden_common_df, bernie_common_df]
biden_common_df

word_freqdist = FreqDist(biden_bank)
most_common_biden = word_freqdist.most_common(100)
biden_common_df = pd.DataFrame(most_common_biden, columns=['word', 'count'])
biden_common_df.set_index('word').to_dict()
biden_common_df = pd.DataFrame({biden_common_df['word']:biden_common_df['count']})
# Create the word cloud:
from wordcloud import WordCloud

wordcloud = WordCloud(colormap='Spectral').generate_from_frequencies(biden_dictionary)

# Display the generated image w/ matplotlib:

plt.figure(figsize=(10,10), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)

# Uncomment the next line if you want to save your image:
plt.savefig('biden_wordcloud.png')

plt.show()



# TF IDF VISUALIZATION
import seaborn as sns
from matplotlib import pyplot as plt

new_figure = plt.figure(figsize=(16,4))

ax = new_figure.add_subplot(121)
ax.title.set_text('TF-IDF Scores')

# Generate a line plot on first axes
color = cm.viridis_r(np.linspace(.4,.8, 30))
ax.bar(tfidf_df['word'][0:20], tfidf_df['tfidf'][0:20], color=color)
# ax.plot(colormap='PRGn')

# Draw a scatter plot on 2nd axes
for ax in new_figure.axes:
    plt.sca(ax)
    plt.xticks(rotation=60)


# plt.savefig('word count bar graphs.png')

plt.savefig('tfidf.png', dpi=400)

amy_tweets_words = []
for i in amy['tweet']:
    amy_tweets_words.append(i)

amy_df1 = pd.DataFrame(amy_tweets_words)
amy_word_freqdist = FreqDist(amy_df1)

wordfreq = []
for w in amy_tweets_words:
    dict = {}
    for j in w.split():
        j = j.lower()
        if j in dict:
            dict[j]+=1
        else:
            dict[j]=1
    wordfreq.append(dict)


total = 230396


ij = []
for i in amy['tweet']:
    i = i.split()
    ij.extend(i)

i_lower = []
for i in ij:
    i = i.lower()
    i_lower.append(i)



dict = {}
for w in i_lower:
    if w in dict:
        dict[w]+=1
    else:
        dict[w]=1


all_tweets_dict = dict
each_tweet_list = wordfreq




import textblob
from textblob import TextBlob
import nltk

scores = []

for i in trump['tweet']:
    blob = TextBlob(i)
    sent = format(blob.sentiment)
    sentimize(sent)


amydf = amy['tweet']
berniedf = bernie['tweet']
bloombergdf = bloomberg['tweet']
bloombergdf1=pd.DataFrame(scores,columns=['polarity','subjectivity'])
buttigiegdf = buttigieg['tweet']
bloombergdf1 = pd.DataFrame(scores,columns=['polarity','subjectivity'])
trumpdf = trump['tweet']
trumpdf1 = pd.DataFrame(scores, columns=['polarity', 'subjectivity'])


trump_op = pd.merge(trumpdf, trumpdf1, left_index=True, right_index=True)







import re
def sentimize(sent):
    p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
    floats = [float(i) for i in p.findall(sent)]  # Convert strings to float
    scores.append(floats)








sns.set_style('darkgrid')
fig = plt.figure(figsize = (20,14))
fig.subplots_adjust(hspace = .30)

ax1 = fig.add_subplot(221)
ax1.hist(trump_op['subjectivity'], bins = 25, label ='Bernie', alpha = .5,edgecolor= 'black',color ='lightblue')
#ax1.hist(bloomberg_op['subjectivity'], bins = 25, label ='Amy', alpha = .3,edgecolor= 'black',color ='grey')

ax1.set_title('Subjectivity of Bernie Tweets')
ax1.legend(loc = 'upper right')
ax1.set(xlim=(0, -1))
plt.savefig('bernie_subjectivity', dpi=400)


trump = pd.read_csv('trump_cleaned.csv')

# #NLPrimaries
### _An analysis of tweets mentioning the 2020 Democratic presidential candidates through the use of natural language processing_
Project Partner: [Julia Chong](https://github.com/juliachong "Julia Chong's GitHub")

## Goal
Analyze tweets _mentioning_ (not written by) the 2020 Democratic presidential candidates, in efforts to see how each candidate is viewed and key topics that are being discussed

## Data Cleaning/Preprocessing
__Thousands of tweets for each candidate__

__[Twint](https://github.com/twintproject/twint "Twint Documentation") - An advanced Twitter scraping tool__
* Make sure [nest-asyncio](https://pypi.org/project/nest-asyncio/ "nest-asyncio Documentation") is installed, imported, and applied
1. Each tweet contains only one candidate's name and/or Twitter handle
2. Tweets are from verified accounts only to contain number of tweets
3. Removed links and images using regular expressions, stopwords, and non-English from tweets using [langdetect](https://pypi.org/project/langdetect/ "langdetect documentation")

## Sentiment Analysis
__[VADER](https://github.com/cjhutto/vaderSentiment "VADER Documentation") (Valence Aware Dictionary and sEntiment Reasoner) Sentiment Analysis - a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media__
* Follow documentation to install, import, and implement
* Get positive, negative, neutral, and compound polarity scores for tweets about each candidate
  <p align="center"><img src="https://github.com/ralterman/NLPrimaries/blob/master/images/sentiment_functions.png"></p>
* Obtain average polarity score in each category for each candidate
* Mark each tweet as either positive or negative depending on which score is higher
* Create dictionary with average number of positive and negative tweets for each candidate to use for visualization
  <p align="center"><img src="https://github.com/ralterman/NLPrimaries/blob/master/images/sentiment.png"></p>

## Subjectivity Analysis
__[TextBlob](https://textblob.readthedocs.io/en/dev/quickstart.html "TextBlob Documentation") - provides access to common text-processing operations__

_Subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective_
* Follow documentation to install, import, and implement
* Obtain sentiment for each tweet
* Use regular expressions to capture float values and then convert these strings to floats
  <p align="center"><img src="https://github.com/ralterman/NLPrimaries/blob/master/images/subjectivity.png"></p>

## TF-IDF — _Term Frequency Inverse Document Frequency_
__Use Scikit-Learn's [Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html "TFIDF Documentation") method__
* Tokenize and Lemmatize all of the tweets first using [NLTK](https://www.nltk.org/ "NLTK Documentation") functions, and make all of the     tokens lowercase
* Remove NLTK's set of stopwords and add our own stopwords to remove
* Convert a collection of text documents to a matrix of token counts
* Use TfidfVectorizer(), TfidfTransformer(), CountVectorizer(), fit_transform(), and fit() functions to get TF-IDF vector for each document (candidate's tweets)
  <p align="center"><img src="https://github.com/ralterman/NLPrimaries/blob/master/images/tfidf.png"></p>

## Word Clouds
__Leverage NLTK [FreqDist()](http://www.nltk.org/api/nltk.html?highlight=freqdist "FreqDist Documentation") function__
* Displays the most commonly used words in tweets about each candidate, where the bigger the word appears in the cloud, the more commonly it is used:
  <p align="center"><img src="https://github.com/ralterman/NLPrimaries/blob/master/images/wordcloudcode.png"></p>
  <p align="center"><img src="https://github.com/ralterman/NLPrimaries/blob/master/images/wordclouds1.png"></p>
  <p align="center"><img src="https://github.com/ralterman/NLPrimaries/blob/master/images/wordclouds2.png"></p>

## Topic Modeling with LDA ([Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation "LDA Wiki"))
__Used [pyLDAvis](https://pyldavis.readthedocs.io/en/latest/readme.html "pyLDAvis Documentation") - Python library for interactive topic model visualization__
* Create one list of all of the tweets across candidates and eliminate given and implemented stopwords
* Run CountVectorizer() and TfidfVectorizer() functions on the tweets, followed by the LatentDirichletAllocation() function
* Create topic models using either TF or TF-IDF

_LDA separates the tokens in the tweets based on underlying, unobserved similar topics to help users interpret them. Check out the demo below:_
  ### [Demo](https://drive.google.com/file/d/1aTVkNNAaKgbUXUCuS0lMgQBpQ16ebItX/view?usp=sharing "LDA Demo")

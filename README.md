# #NLPrimaries
### _An analysis of tweets mentioning the 2020 Democratic presidential candidates through the use of natural language processing_
Project Partner: [Julia Chong](https://github.com/juliachong "Julia Chong's GitHub")

## Goal
Analyze tweets _mentioning_ (not written by) the 2020 Democratic presidential candidates, in efforts to see how each candidate is viewed and key topics that are being discussed

## Data Cleaning/Preprocessing
__[Twint](https://github.com/twintproject/twint "Twint Documentation") - An advanced Twitter scraping tool__
* Make sure [nest-asyncio](https://pypi.org/project/nest-asyncio/ "nest-asyncio Documentation") is installed, imported, and applied

__Thousands of tweets for each candidate__
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

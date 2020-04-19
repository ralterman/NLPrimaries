# #NLPrimaries
### _An analysis of tweets mentioning the 2020 Democratic presidential candidates through the use of natural language processing_
Project Partner: [Julia Chong](https://github.com/juliachong "Julia Chong's GitHub")

## Goal
Analyze tweets _mentioning_ the 2020 Democratic presidential candidates, in efforts to see how each candidate is viewed and key topics that are being discussed

## Data Cleaning/Preprocessing
__[Twint](https://github.com/twintproject/twint "Twint Documentation") - An advanced Twitter scraping tool__
* Make sure [nest-asyncio](https://pypi.org/project/nest-asyncio/ "nest-asyncio Documentation") is installed, imported, and applied
__Thousands of tweets for each candidate__
1. Each tweet contains only one candidate's name and/or Twitter handle
2. Tweets are from verified accounts only to contain number of tweets
3. Removed links, images, stopwords, and non-English from tweets

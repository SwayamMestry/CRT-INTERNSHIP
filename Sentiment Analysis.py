#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')


# In[2]:


customer_reviews = [
    "Very good product! It exceeded my expectations.",
    "The customer service was bad, and the product arrived damaged.",
    "The user interface is new but easy to use.",
    "I'm happy with the quality of the product.",
    "This company provides poor value for the price."
]


# In[3]:


sia = SentimentIntensityAnalyzer()


# In[4]:


sentiments = []
for review in customer_reviews:
    sentiment_scores = sia.polarity_scores(review)
    sentiment_label = "Positive" if sentiment_scores['compound'] > 0 else "Negative" if sentiment_scores['compound'] < 0 else "Neutral"
    sentiments.append(sentiment_label)


# In[5]:


import pandas as pd


# In[6]:


data = {'Review': customer_reviews, 'Sentiment': sentiments}
df = pd.DataFrame(data)


# In[7]:


df


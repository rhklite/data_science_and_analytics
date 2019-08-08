#!/usr/bin/env python
# coding: utf-8

# # MIE 1624 Assignment 1: Sentiment Analysis
# 
# Answer the question: **What can public opinion on Twitter tell us about the US airline in 2015?**

# In[1]:


# !pip3 install pandas numpy --user
# !pip3 install --user
# !pip3 install HTMLParser --user
# !pip3 install nltk --user

import pandas as pd
import numpy as np
from html.parser import HTMLParser
from bs4 import BeautifulSoup
import unicodedata
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import sys
print(sys.version) # showing current python version


# ## Part 1: Data Cleaning

# ### 1.1 Reading Files

# In[ ]:


generic_tweets = pd.read_csv('generic_tweets.txt', parse_dates=['date'])
generic_tweets.head()


# In[ ]:


airline_tweets = pd.read_csv('US_airline_tweets.csv')
airline_tweets.head()


# In[ ]:


colnames=['word', 'score'] # defining column names for the file
corpus=pd.read_csv('corpus.txt', names=colnames, delim_whitespace=True) #read in the file, give columns a name, use whitespace as deimiter
corpus.head()


# In[ ]:


colnames=['word']
stop_words=pd.read_csv('stop_words.txt',names=colnames) #read in the file, give column a name
stop_words.head()


# ### 1.2 [Remove HTML tags and attributes](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) and return HTML character codes (i.e., &...;) with an ASCII equivalent

# In[ ]:


# defining function to be applied to all tweet
def parse(text):
    """
    remove html tags and attributes using beautifulSoup html.parser
    returns output as text
    """
    soup=BeautifulSoup(text,'html.parser')
    return soup.get_text()


# In[ ]:


generic_tweets['text']=generic_tweets['text'].apply(parse)
generic_tweets.head()


# In[ ]:


airline_tweets['text']=airline_tweets['text'].apply(parse)
airline_tweets.head()


# ### 1.3 [Normalize Unicode character with regular text](https://stackoverflow.com/a/3194567/10251025)

# In[ ]:


def normalize(text):
    """
    normalizes unicode character to regular text
    Example: normalize('Cześć')='Czesc'
    """
    # read ascii characters using NFKD method, then decode back to string
    text=unicodedata.normalize('NFKD',text).encode('ascii', 'ignore').decode('utf-8')
    return text


# In[ ]:


generic_tweets['text']=generic_tweets['text'].apply(normalize)
airline_tweets['text']=airline_tweets['text'].apply(normalize)


# ### 1.4 [Removing URL](https://docs.python.org/2/library/re.html)

# In[ ]:


def remove_URL(text):
    """
    removes URL within a string by matching a pattern
    pattern: http or www until 1st white space seen
    """
    try:
        text = re.sub(r'http\S+\s+', '', text) #match pattern http followed by any non whitespace followed by white space, replace with nothing
        text = re.sub(r'www\S+\s+', '', text)#match pattern www followed by any non whitespace followed by white space, replace with nothing
    except:
        text = text
    return(text)


# In[ ]:


generic_tweets['text']=generic_tweets['text'].apply(remove_URL)
airline_tweets['text']=airline_tweets['text'].apply(remove_URL)


# ### 1.5 Removing non letter and white spaces

# In[ ]:


def remove_ws(text):
    """
    Removing extra whitespace, trailing and leading white spaces
    """
    try:
        text=re.sub(r'\s+', ' ', text) #removes extra white space, strip leading and trailing white space
        text=text.lower() #make all text lower case
    except:
        text=text
    return text


# In[ ]:


def remove_nonletter(text):
    """
    removes non a-z or A-Z letters
    """
    try:
        text=re.sub(r'[^a-zA-Z ]+', ' ', text) #matching anything that is not in the a-z, A-Z and white space set, replace with white space
    except:
        text=text
    return text


# In[ ]:


#removing white space and non letter from generic_tweets dataframe
generic_tweets['text']=generic_tweets['text'].apply(remove_nonletter)
generic_tweets['text']=generic_tweets['text'].apply(remove_ws)


# In[ ]:


generic_tweets.head()


# ### 1.6 [Storing airline mentions](https://stackoverflow.com/a/43958095/10251025)

# In[ ]:


#removing white space from airline_tweets dataframe
airline_tweets['text']=airline_tweets['text'].apply(remove_ws)


# In[ ]:


#define a tuple of airline names. This is to find the actual airline mention within a tweet
airlines={'@virginamerica':'virginamerica', '@united':'united', '@southwestair':'southwestair', '@jetblue':'jetblue','@usairways':'usairways', '@americanair':'americanair', '@deltaassist':'deltaassist'}

def find_mentions(text):
    global airlines
    text=re.findall(r'@\S+',text)
    if len(text)>=1:
        for i in text:
            for j in airlines:
                if j in i:return airlines[j]


# In[ ]:


airline_tweets['airline']=airline_tweets['text'].apply(find_mentions)
airline_tweets['text']=airline_tweets['text'].apply(remove_nonletter)
airline_tweets['airline'].unique()


# In[ ]:


airline_tweets.head()


# In[ ]:


airline_tweets['airline'].unique()


# In[ ]:


airline_tweets[airline_tweets['airline'].isnull()]


# In[ ]:


# removing the tweet where Jetblue replied to a user
airline_tweets=airline_tweets.dropna(subset=['airline'])
airline_tweets[airline_tweets['airline'].isnull()]


# ### 1.7 Remove stop words

# In[ ]:


# getting ntlk stop word list into dataframe
stop_words_ntlk=pd.DataFrame(data=list(stopwords.words('english')), columns=['word'])
print(len(stop_words_ntlk))
stop_words_ntlk.head()


# In[ ]:


#considering the names of the airlines as stopwords aswell since they do not add any value to the analysis
airlines_values=pd.DataFrame(data=list(airlines.values()), columns=['word'])
airlines_values.head()


# In[ ]:


# combining given stop_word list, ntlk stop_word list and airline names
# stop_words_combined=stop_words.append([stop_words_ntlk, airlines_values], ignore_index=True).drop_duplicates().reset_index(drop=True)
stop_words_combined=stop_words_ntlk
print(len(stop_words_combined))
stop_words_combined.tail()


# In[ ]:


# tokenizing the text and removing stop words for airline tweets
airline_tweets['text']=airline_tweets['text'].apply(lambda x : x.split())
airline_tweets['text']=airline_tweets['text'].apply(lambda y : [word for word in y if word not in stop_words_combined.values])
airline_tweets.head()


# In[ ]:


# tokenizing the text and removing stop words for generic tweets
generic_tweets['text']=generic_tweets['text'].apply(lambda x : x.split())
generic_tweets['text']=generic_tweets['text'].apply(lambda y : [word for word in y if word not in stop_words_combined.values])
generic_tweets.head()


# ### 1.8 Lemmatising

# In[ ]:


# ## construct stemmer
# from nltk.stem import PorterStemmer
# stemmer=PorterStemmer()

# def stemming(text):
#     x = [stemmer.stem(word) for word in text]
#     return x

# generic_tweets['text']=generic_tweets.text.apply(stemming)


# In[ ]:


# construct lemmatising

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatising (text):
    x = [lemmatizer.lemmatize(word) for word in text]
    return x

generic_tweets['text']=generic_tweets.text.apply(lemmatising)


# In[ ]:


# airline_tweets['text']=airline_tweets.text.apply(stemming)
airline_tweets['text']=airline_tweets.text.apply(lemmatising)


# ## Part 2: Exploratory Analysis

# ### 2.1 Extract airline mentions
# 
# 
# this is already done in step 1.6. Was done before removing the stop words since the name of the airline doesn't add sentiment to the analysis. Below is a visualization of the number of mentions of each airline.

# In[ ]:


print(airline_tweets['airline'].value_counts(normalize=True));print('\n')
print(airline_tweets['airline'].value_counts());print('\n')
print('Total Tweet Count: '+ str(airline_tweets['airline'].value_counts().values.sum()))


# In[ ]:


#showing all airline tweets that sentiment column has the value 'negative' and group by the airline column ...
#...showing only sentiment column, and count them
tmp=airline_tweets[airline_tweets.sentiment=='negative'].groupby('airline').sentiment.agg(['count'])
tmp.columns=['negative']
tmp2=airline_tweets[airline_tweets.sentiment=='positive'].groupby('airline').sentiment.agg(['count'])
tmp2.columns=['positive']
tmp=tmp.join(tmp2).fillna(0)
tmp['%neg']=tmp.apply(lambda x: int(x['negative'])/(int(x['positive'])+int(x['negative']))*100,axis=1)
tmp['%pos']=tmp.apply(lambda x: int(x['positive'])/(int(x['positive'])+int(x['negative']))*100,axis=1)
tmp['total']=tmp.apply(lambda x: (int(x['positive'])+int(x['negative'])),axis=1)
tmp=tmp.sort_values(by=['total'], ascending=False)
tmp2=None
tmp
# print(tmp);print('\n')


# **Observation**
# 
# There are total of around 11540 tweets that mentioned an airline. 
# 
# Of these tweets, [United](https://twitter.com/united?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) airline has the most mentions at 3124 (27.07%) mentions, and [Delta Assist](https://twitter.com/deltaassist?lang=en) (which refers to delta airlines) has the least mentions at 2, which is 0.02% of total number of mentions. 
# 
# [US airways](https://twitter.com/USAirways?lang=en) and [American Air](https://twitter.com/AmericanAir?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor) has similar number of mentions, at 2532 (21.94%) and 2296 (19.90%) respectively.
# 
# United, US Airways and American air all recieved similar percentages of negative comments, around 84~89% range, and virginamerica recieved the most percentage of positive comments, at 45.6%.

# ### 2.2 Dataset Visualization

# #### Plot for US Airline Tweets

# In[ ]:


labels = list(airline_tweets['airline'].value_counts().keys())
sizes = list(airline_tweets['airline'].value_counts().values)
explode = (0, 0, 0, 0, 0, 0.5, 1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%', startangle=90, shadow=True)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
fig = plt.gcf()
fig.set_size_inches(10,10) # or (4,4) or (5,5) or whatever
plt.title('US Airline Twitter Mention Distribution', fontsize=16)
plt.show()


# In[ ]:


#airline_tweets.groupby(['airline','sentiment']).sentiment.agg(['count'])


# In[ ]:


labels = list(tmp.index.values)

neg = (list(tmp.negative.values))
pos = (list(tmp.positive.values))
ind = np.arange(len(labels))    # the x locations for the groups
width = 0.6     # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, neg, width)
p2 = plt.bar(ind, pos, width, bottom=neg)

plt.ylabel('Number of Tweets')
plt.xlabel('US Airlines')
plt.title('Number of Tweets of each Airline', fontsize=16)
plt.xticks(ind, labels)
# plt.yticks(np.arange(0, 81, 10))
plt.xticks(rotation=-75)
plt.legend((p1[0], p2[0]), ('Negative', 'Positive'))

plt.show()


# #### Plot for Generic Tweets

# In[ ]:


generic_tweets['length'] = generic_tweets['text'].apply(len)
generic_tweets.head()


# In[ ]:


generic_tweets[generic_tweets['class'] == 1]['length'].values


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
generic_tweets.hist('length', by='class', ax=axes)
plt.suptitle('Length of Tweet for Positive/Negative Sentiment', x=0.5, y=1.05, ha='center', fontsize='xx-large')
fig.text(0.5, 0.04, 'Length Bin', ha='center')
fig.text(-0.04, 0.5, 'Frequency', va='center', rotation='vertical')


# In[ ]:


generic_tweets.hist('class')


# ## 3. Model Preparation

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[ ]:


def make_str(text):
    y=''
    for x in text:
        y= y+' '+x
    return y


# **Model Prep Starts Here**

# In[ ]:


generic_tweets_vec=generic_tweets.copy()
generic_tweets_vec.text=generic_tweets_vec.text.apply(make_str)

# Create a new column called df.elderly where the value is yes
# if df.age is greater than 50 and no if not
generic_tweets_vec['sentiment'] = np.where(generic_tweets_vec['class']==4, 'positive', 'negative')


# In[ ]:


cv=CountVectorizer()
tfidf = TfidfTransformer()

generic_bag=cv.fit_transform(generic_tweets_vec.text)
generic_trans = tfidf.fit_transform(generic_bag)

generic_trans.shape


# In[ ]:


X=generic_trans
Y=generic_tweets_vec['sentiment']

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.3)


# In[ ]:


model=LogisticRegression()

model.fit(X_train,Y_train)


# In[ ]:


y_pred=model.predict(X_test)


# In[ ]:


import seaborn as sns
from sklearn import metrics
cm=metrics.confusion_matrix(Y_test, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="", linewidths=.5, square = True, cmap = 'Blues');
plt.ylabel('Actual label', size = 12);
plt.xlabel('Predicted label', size = 12);
all_sample_title = 'Generic Tweets Confusion Matrix'
plt.title(all_sample_title, size = 16);


# In[ ]:


print ('Accuracy Score: ' + str(round(accuracy_score(Y_test, y_pred),3)))
print ('\n')
print(classification_report(Y_test,y_pred))


# ## 4. Model Implementation

# ### 4.1 Eavluating Airline Model with Generic Tweets Model

# In[ ]:


airline_tweets_vec=airline_tweets.copy()
airline_tweets_vec.text=airline_tweets_vec.text.apply(make_str)


# In[ ]:


airline_bag=cv.transform(airline_tweets_vec.text)
airline_trans = tfidf.transform(airline_bag)

airline_trans.shape


# In[ ]:


y_pred_air=model.predict(airline_trans)


# In[ ]:


y_pred_air


# In[ ]:


print (round(accuracy_score(airline_tweets_vec.sentiment, y_pred_air),3))

print(classification_report(airline_tweets_vec.sentiment, y_pred_air))


# ### 4.2 Training a Airline Logistic Regression Model

# In[ ]:


model_airline=LogisticRegression()


# In[ ]:


airline_bag=cv.fit_transform(airline_tweets_vec.text)
airline_fit = TfidfTransformer().fit(airline_bag)
airline_trans = airline_fit.transform(airline_bag)
airline_trans.shape


# In[ ]:


X_air=airline_trans
Y_air=airline_tweets_vec['sentiment']

X_train_air, X_test_air, Y_train_air, Y_test_air=train_test_split(X_air,Y_air,test_size=0.3)


# In[ ]:


model_airline.fit(X_train_air,Y_train_air)


# In[ ]:


y_pred_air=model_airline.predict(X_test_air)


# In[ ]:


import seaborn as sns
from sklearn import metrics
cm=metrics.confusion_matrix(Y_test_air, y_pred_air)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="", linewidths=.5, square = True, cmap = 'Blues');
plt.ylabel('Actual label', size = 12);
plt.xlabel('Predicted label', size = 12);
all_sample_title = 'Generic Tweets Confusion Matrix'
plt.title(all_sample_title, size = 16);


print (round(accuracy_score(Y_test_air, y_pred_air),3))

print(classification_report(Y_test_air,y_pred_air))


# ### 4.3 Airline Multi-Class logistic regression model

# In[ ]:


airline_tweets_vec.negative_reason.unique()


# In[ ]:


airline_tweets_vec.negative_reason.unique()


# In[ ]:


airline_tweets_neg=airline_tweets_vec[airline_tweets_vec.sentiment=='negative']
airline_tweets_neg.head()


# In[ ]:


cv=CountVectorizer()
tfidf = TfidfTransformer()

airline_tweets_neg=cv.fit_transform(airline_tweets_neg.text)
airline_neg_trans = tfidf.fit_transform(airline_tweets_neg)

airline_neg_trans.shape


# In[ ]:


X_multi=airline_neg_trans
Y_multi=airline_tweets_vec[airline_tweets_vec.sentiment=='negative'].negative_reason

X_multi_train, X_multi_test, Y_multi_train, Y_multi_test = train_test_split(X_multi,Y_multi,test_size=0.3, random_state=101)


# In[ ]:


model_multi = LogisticRegression()


# In[ ]:


model_multi.fit(X_multi_train, Y_multi_train)


# In[ ]:


y_multi_predict = model_multi.predict(X_multi_test)


# In[ ]:


resultsDF = pd.DataFrame({
        'true':Y_multi_test,
        'predicted':y_multi_predict
    })
resultsDF.head()


# In[ ]:


cm=metrics.confusion_matrix(Y_multi_test, y_multi_predict, labels=airline_tweets_vec.negative_reason.unique())

plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, fmt="", linewidths=.5, square = True, cmap = 'Blues');
plt.ylabel('Actual label', size = 12);
plt.xlabel('Predicted label', size = 12);
all_sample_title = 'Generic Tweets Confusion Matrix'
plt.title(all_sample_title, size = 16);


classes = airline_tweets_vec.negative_reason.unique()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

print (round(accuracy_score(Y_multi_test, y_multi_predict),3))

print(classification_report(Y_multi_test,y_multi_predict))


# In[ ]:





# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:10:51 2019

@author: Sameer Misbah
"""

"""

Word positive negative bad
xxx   1/0     1/0      1/0

"""

"""
Resources
https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
'Multinomial: It is used for discrete counts. For example, let’s say,  we have a text classification problem. Here we can consider bernoulli trials which is one step further and instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as “number of times outcome number x_i is observed over the n trials”.'
http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf
https://data.world/datasets/twitter
"""



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import pandas as pd


Positive = 0
Negative = 0
Neutral = 0
Bad = 0



# function to calculate percentage
def percentage(part, whole):
    return 100 * float(part) / float(whole)


def getStringArrayFromNumpyDataFrame(dataframe):
    list=[]
    for s in dataframe.values:
        if len(str(s[0]))>0:
            list.append(str(s[0]))
    return list

def getEmotions(text,clf):

    text=text.split()
    X_new_counts = count_vect.transform(text)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    predicted = clf.predict(X_new_tfidf)
    positive_count=0
    for x in predicted:
        if x==1:
            positive_count=positive_count+1
    return positive_count;


def getEmotionFromText(text):
    global Positive
    global Negative
    global Neutral
    global Bad
    positives=getEmotions(text,clf_positive)
    negatives=getEmotions(text,clf_negative)
    bad=getEmotions(text,clf_bad)
    if(bad>0):
        Bad += 1
        return("Bad tweet ->%s"%(text))
    else:
        if(positives-negatives)>0:
            Positive += 1
            return("Positive tweet ->%s"%(text))
        elif(negatives-positives)>0:
            Negative += 1
            return("Negative tweet ->%s"%(text))
        else:
            Neutral += 1
            return("Neutral tweet ->%s"%(text))


train_data_csv_name="TrumpWords.csv"

df_x_words = pd.read_csv(train_data_csv_name,usecols=[0], encoding="utf-8")
df_y_positive= pd.read_csv(train_data_csv_name,usecols=[1], encoding="utf-8")
df_y_negative= pd.read_csv(train_data_csv_name,usecols=[2], encoding="utf-8")
df_y_bad= pd.read_csv(train_data_csv_name,usecols=[3], encoding="utf-8")



count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(getStringArrayFromNumpyDataFrame(df_x_words))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


clf_positive = MultinomialNB().fit(X_train_tfidf, df_y_positive)
clf_negative = MultinomialNB().fit(X_train_tfidf, df_y_negative)
clf_bad = MultinomialNB().fit(X_train_tfidf, df_y_bad)

outputfile = open('SentimentAnalysis.txt','w')

tweet_file_name="TrumpTweets.csv"

with open(tweet_file_name, encoding="utf-8", errors = 'ignore') as f:
    for line in f:
        try:
            tmpstr=getEmotionFromText(line)
            print(tmpstr)
            outputfile.write(tmpstr)
        except:
            pass
        
outputfile.close()


print("\n Positive Tweets Count : ", Positive)
print("\n Nagative Tweets Count : ", Negative)
print("\n Neutal Tweets Count : ", Neutral)
print("\n Bad Tweets Count : ", Bad)


Positive = percentage(Positive, 30385)
Negative = percentage(Negative, 30385)
Neutral = percentage(Neutral, 30385)
Bad = percentage(Bad, 30385)


Positive = format(Positive, '.2f')
Negative = format(Negative, '.2f')
Neutral = format(Neutral, '.2f')
Bad = format(Bad, '.2f')



labels = ['Positive Tweets [' + str(Positive) + '%]', 'Negative Tweets [' + str(Negative) + '%]','Neutral Tweets [' + str(Neutral) + '%]', 'Bad Tweets [' + str(Bad) + '%]']
sizes = [Positive, Negative, Neutral, Bad]
colors = ['yellowgreen','lightgreen','darkgreen', 'gold']
# 'red','lightsalmon','darkred'
patches, texts = plt.pie(sizes, colors=colors, startangle=90)
plt.legend(patches, labels, loc="best")
plt.title("Sentiment analysis on Trump's tweets by analyzing " + str(30385) + ' Tweets.')
plt.axis('equal')
plt.tight_layout()
plt.show()
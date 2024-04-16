import nltk  as nltk
from nltk.corpus import stopwords
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

movies_data=pd.read_csv('m.csv')
m=movies_data['text'].values.astype(str)
m1=np.array_str(m)
stop=set(stopwords.words("english"))
words=m1.split()

final=[]
for w in words:
    if w not in final:
        final.append(w)
p=dict()
n=dict()
p1=[]
n1=[]
sentiment_analyzer=SentimentIntensityAnalyzer()
for i in words:
    if not i.lower() in stop:
        polarity=sentiment_analyzer.polarity_scores(i)
        if (polarity['compound']>=0.05):
            p[i]=polarity['compound']
        if (polarity['compound']<=0.05):
            n[i]=polarity['compound']

print(p)
print(n)


word_cloud=WordCloud(collocations=False).generate(m1)
plt.figure()
plt.imshow(word_cloud,interpolation="bilinear")
plt.axis("off")
plt.show()

import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
from nltk.tokenize  import word_tokenize,sent_tokenize

te="So, keep working. Keep striving. Never give up. Fall down seven times, get up eight. Ease is a greater threat to progress than hardship. Ease is a greater threat to progress than hardship. So, keep moving, keep growing, keep learning. See you at work."
te=re.sub(r'\[[0-9]*\]', ' ', te)
te=re.sub(r'\s+', ' ', te)
te=re.sub(r'\[[0-9]{}*\]' ,' ', te)
ft=re.sub('[^a-zA-Z]', ' ' ,te)
print("\n Text after removing Special char , symbols and digits \n")
print(ft)
print("\n")
sw=set(stopwords.words("english"))
print("\n Stopwords..... \n")
print(sw)
wd=word_tokenize(ft)
print("\n Word Tokens..... \n")
print(wd)
print("\n Extractive Text Summery \n ")
wordfreq = {}
for word in wd:
  if word in sw:
      continue
  if word in wordfreq:
     wordfreq[word] += 1
  else:
    wordfreq[word] = 1
maximum_frequency = max(wordfreq.values())
for word in wordfreq.keys():
    wordfreq[word] = (wordfreq[word]/maximum_frequency)
sentences = sent_tokenize(ft)
sentenceValue = {}
for sentence in sentences:
   for word, freq in wordfreq.items():
       if word in sentence.lower():
          if sentence in sentenceValue:
               sentenceValue[sentence] += freq
          else:
              sentenceValue[sentence] = freq
import heapq
summary = ''
summary_sentences = heapq.nlargest(4, sentenceValue, key=sentenceValue.get)
summary = ' '.join(summary_sentences)
print(summary)
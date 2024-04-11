# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# import re
# from nltk.corpus import stopwords
# from nltk.tokenize  import word_tokenize,sent_tokenize

# te=" Text semmerisation is an nlp technique that extract text from a large amount of data. It is the process of identifying the most meanigful information in a document and compressing it into a shorter version by preserving its meaning. @ $ Types 12 : Extractive Text Summerisation "
# te=re.sub(r'\[[0-9]*\]', ' ', te)
# te=re.sub(r'\s+', ' ', te)
# te=re.sub(r'\[[0-9]{}*\]' ,' ', te)
# ft=re.sub('[^a-zA-Z]', ' ' ,te)
# print("\n Text after removing Special char , symbols and digits \n")
# print(ft)
# print("\n")
# sw=set(stopwords.words("english"))
# print("\n Stopwords..... \n")
# print(sw)
# wd=word_tokenize(ft)
# print("\n Word Tokens..... \n")
# print(wd)
# print("\n Extractive Text Summery \n ")
# wordfreq = {}
# for word in wd:
#   if word in sw:
#       continue
#   if word in wordfreq:
#      wordfreq[word] += 1
#   else:
#     wordfreq[word] = 1
# maximum_frequency = max(wordfreq.values())
# for word in wordfreq.keys():
#     wordfreq[word] = (wordfreq[word]/maximum_frequency)
# sentences = sent_tokenize(ft)
# sentenceValue = {}
# for sentence in sentences:
#    for word, freq in wordfreq.items():
#        if word in sentence.lower():
#           if sentence in sentenceValue:
#                sentenceValue[sentence] += freq
#           else:
#               sentenceValue[sentence] = freq
# import heapq
# summary = ''
# summary_sentences = heapq.nlargest(4, sentenceValue, key=sentenceValue.get)
# summary = ' '.join(summary_sentences)
# print(summary)



import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Input text
te = "Text semmerisation is an nlp technique that extract text from a large amount of data. It is the process of identifying the most meanigful information in a document and compressing it into a shorter version by preserving its meaning. @ $ Types 12 : Extractive Text Summerisation"

# Remove square brackets and extra whitespaces
te = re.sub(r'\[[0-9]*\]', ' ', te)
te = re.sub(r'\s+', ' ', te)
te = re.sub(r'\[[0-9]{}*\]' ,' ', te)

# Remove non-alphabetic characters
ft = re.sub('[^a-zA-Z]', ' ' ,te)
print("\nText after removing Special char , symbols and digits \n")
print(ft)
print("\n")

# Set of English stopwords
sw = set(stopwords.words("english"))
print("\nStopwords..... \n")
print(sw)

# Tokenize words
wd = word_tokenize(ft)
print("\nWord Tokens..... \n")
print(wd)

print("\nExtractive Text Summary\n")
# Calculate word frequencies
wordfreq = {}
for word in wd:
    if word in sw:
        continue
    if word in wordfreq:
        wordfreq[word] += 1
    else:
        wordfreq[word] = 1

# Find maximum word frequency
maximum_frequency = max(wordfreq.values())

# Normalize word frequencies
for word in wordfreq.keys():
    wordfreq[word] = (wordfreq[word] / maximum_frequency)

# Tokenize sentences
sentences = sent_tokenize(ft)

# Calculate sentence scores based on word frequencies
sentenceValue = {}
for sentence in sentences:
    for word, freq in wordfreq.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

# Select top 4 sentences with highest scores
import heapq
summary_sentences = heapq.nlargest(4, sentenceValue, key=sentenceValue.get)
summary = ' '.join(summary_sentences)
print(summary)

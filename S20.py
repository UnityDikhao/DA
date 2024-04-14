import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
from nltk.tokenize  import word_tokenize,sent_tokenize
from nltk.probability import FreqDist

te="""Hello all, Welcome to Python Programming Academy. Python Programming Academy is a nice platform to learn new programming skills. It is difficult to get enrolled in this Academy so."""
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
st=sent_tokenize(te)
print("\n Sent Tokens..... \n")
print(st)
wd=word_tokenize(te)
print("\n Word Tokens..... \n")
print(wd)
ft=[]
for w in wd:
    if w not in sw:
        ft.append(w)
print("\n Text After removing Stopwords \n ",ft)

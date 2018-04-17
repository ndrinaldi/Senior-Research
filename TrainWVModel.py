import os, io, sys, re
import sqlite3
import gensim
from stemming.porter2 import stem

stop_list = ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
 "in", "is", "it", "its", "of", "on", "than", "that", "the", "to", "was", "were",
  "will", "with", "which"]

full_stop_abbv = {"!":".", "?":".", "-":" ", "U.S.":"US", "Co.":"Co", "Corp.":"Corp",
 "a.m.": "am", "p.m.":"pm", "Inc.":"Inc", "Jan.": "Jan", "Feb.":"Feb", "Mar.":"Mar",
 "Apr.":"Apr", "Jun.":"Jun", "Jul.":"Jul", "Aug.":"Aug", "Sep.":"Sept", "Sept.":"Sept",
 "Oct.":"Oct", "Nov.":"Nov", "Dec.":"Dec"}

ReutersRoot = '/Users/loaner/Desktop/ReutersNews106521'
BloomRoot = '/Users/loaner/Desktop/20061020_20131126_bloomberg_news'
BloomSample = '/Users/loaner/Desktop/SampleBloom'
ReutersSample = '/Users/loaner/Desktop/SampleReuters'



raw_f = open("raw_articles.txt","w")

# Database of articles

database = "finance_corpus.db"
conn = sqlite3.connect(database)
c = conn.cursor()
c.execute("SELECT body FROM articles")
rows = c.fetchall()
for article in rows:
    raw_f.write(article[0])

# Articles from headline database

tempBloom = []
for root, dirs, files in os.walk(BloomRoot):
    for file_ in files:
        count = 0
        article = ""
        f_ = io.open(os.path.join(root, file_), 'r', encoding="Latin-1")
        for line in f_:
            if (count > 6 and line != "\n"):
                article += line[:-1]+' '
            count += 1
        tempBloom.append(article+'\n')
for i in range(1, len(tempBloom), 1):
    raw_f.write(tempBloom[i])

tempRueters = []
for root, dirs, files in os.walk(ReutersRoot):
    for file_ in files:
        count = 0
        article = ""
        f_ = io.open(os.path.join(root, file_), 'r', encoding="Latin-1")
        for line in f_:
            if count > 8:
                article += line[:-1]+' '
            count += 1
        tempRueters.append(article+'\n')
for i in range(1, len(tempRueters), 1):
    raw_f.write(tempRueters[i])


raw_f.close()


raw_f = open("raw_articles.txt","r")
articles = raw_f.read().split('\n')

t_data = []

for article in articles:
    article = article.replace('-', ' ')                 # Replace - with space
    article = re.sub('[^A-Za-z0-9\s,.]+', '', article)  # Remove special characters
    for key, value in full_stop_abbv.items():           # Replace all special terminals
        article = article.replace(key, value)           # and full stop abbriviations

    for sentence in article.split(". "):
        t_element = [].
        for word in sentence.split():
            word = word.lower()                         # Only lower case
            #if (word not in stop_list):   (halted)     # Remove stop words
            t_element.append(word)

        if t_element:                                   # Remove empties
            t_data.append(t_element)


raw_f.close()

model = gensim.models.Word2Vec(t_data, size=100, window=5, min_count=5)
print(model.wv.most_similar(positive=["sell"]))
print()
print(model.wv.most_similar(positive=["buy"]))
print()
print(model.wv.most_similar(positive=["china"]))

model.save("NLPmodel")

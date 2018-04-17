import os, io, sys, re
import requests
import operator
import gensim
import numpy

full_stop_abbv = {"!":".", "?":".", "-":" ", "U.S.":"US", "Co.":"Co", "Corp.":"Corp",
 "a.m.": "am", "p.m.":"pm", "Inc.":"Inc", "Jan.": "Jan", "Feb.":"Feb", "Mar.":"Mar",
 "Apr.":"Apr", "Jun.":"Jun", "Jul.":"Jul", "Aug.":"Aug", "Sep.":"Sept", "Sept.":"Sept",
 "Oct.":"Oct", "Nov.":"Nov", "Dec.":"Dec"}

# Load Word2Vec model
model = gensim.models.Word2Vec.load("NLPmodel")

ReutersRoot = '/Users/loaner/Desktop/ReutersNews106521'
BloomRoot = '/Users/loaner/Desktop/20061020_20131126_bloomberg_news'
BloomSample = '/Users/loaner/Desktop/SampleBloom'
ReutersSample = '/Users/loaner/Desktop/SampleReuters'

raw_data = []                   # Raw headlines
event_triples = []              # Text event triples with date
embedded_events = []            # Vector triples of the event
event_vectors = []              # Single vectors of events
day_vectors = []                # Single vectors rep. all event in one day

cannot_extract = []             # Work in progress
cannot_embed = []               # text events that could not be vectorized

# Gather and fix headlines from Bloomburg
for root, dirs, files in os.walk(BloomRoot):
    for file_ in files:
        f = io.open(os.path.join(root, file_), 'r', encoding="Latin-1")
        headline = f.readline()[2:-1]
        date = root[-10:]
        # Cleaning up the headline
        if headline != ' ' and headline != '':
            headline = headline.encode('utf8').decode('utf8')
            headline = headline.replace('-', ' ')
            headline = re.sub('[^A-Za-z0-9\s,.]+', '', headline)
            for key, value in full_stop_abbv.items():
                headline = headline.replace(key, value)
            headline += '.'
            raw_data.append((headline, date))
raw_data = raw_data[1:]

# Gather and fix headlines from Reuters
for root, dirs, files in os.walk(ReutersRoot):
   for file_ in files:
       f = io.open(os.path.join(root, file_), 'r', encoding="Latin-1")
       headline = f.readline()[2:-1]
       date = root[-8:-4]+'-'+root[-4:-2]+'-'+root[-2:]
       # Cleaning up the headline
       if headline != ' ' and headline != '':
            headline = headline.encode('utf8').decode('utf8')
            headline = headline.replace('-', ' ')
            headline = re.sub('[^A-Za-z0-9\s,.]+', '', headline)
            for key, value in full_stop_abbv.items():
                headline = headline.replace(key, value)
            headline += '.'
            raw_data.append((headline, date))

# Sort headlines by date
raw_data = sorted(raw_data, key=operator.itemgetter(1))

# Write headlines to a file
f_ = open("Headlines.txt","w")
for line in raw_data:
    f_.write(line[0] + '\n')
f_.close()

# Run ReVerb on clean headlines
os.system("java -Xmx512m -jar reverb-latest.jar Headlines.txt > ReVerbInfo.txt")

# Analyze ReVerb data in event triples
g_ = open("ReVerbInfo.txt", "r")
for line in g_:
    temp = re.split(r'\t+', line)
    event_triples.append((temp[15], temp[16], temp[17][:-1], raw_data[int(temp[1])][1]))
g_.close

# Generate event embeddings
for event in event_triples:
    event_embeddable = True
    temp = []
    for i in range(3):
        arg_embedded = []
        arg = event[i]
        for word in arg.split():
            if word in model.wv.vocab or word == '#':
                if word != '#':
                    arg_embedded.append(model.wv[word])
                #else:
                    #print("found number")  impliment number recognition here
            else:
                event_embeddable = False
                cannot_embed.append(word)
                #print("found no embedding for ", word, " on ", event[3])
                #print(event[0], " ", event[1], " ", event[2])
        if event_embeddable:
            np_array = numpy.array(arg_embedded)
            arg_avg = numpy.mean(np_array, axis=0)
            if(type(arg_avg) is not numpy.ndarray):
                event_embeddable = False
            temp.append(arg_avg.tolist())
    if event_embeddable:
        embedded_events.append((temp[0], temp[1], temp[2], event[3]))
# Un-embeddable words
#print(cannot_embed)

#=====================================================================#

# Here is where a neural-tensor-network should be used to
# embed the event triple into a single vector while retaining
# the relationship between the arguements
for event in embedded_events:
    one_vec = numpy.array([event[0], event[1], event[2]])
    one_vec = numpy.mean(one_vec, axis=0)
    event_vectors.append((one_vec, event[3]))

#=====================================================================#

# Average events of one day into single vectors (expected)
day_vec = [event_vectors[0][0]]
print(day_vec)
for i in range(1, len(event_vectors), 1):
    if (event_vectors[i][1] != event_vectors[i-1][1]):
        temp_np_array = numpy.array(day_vec)
        temp_mean = numpy.mean(temp_np_array, axis=0)
        day_vectors.append((temp_mean, event_vectors[i-1][1]))
        day_vec = []
    day_vec.append(event_vectors[i][0])

numpy.save("day_vectors", numpy.array(day_vectors))

import requests
import operator
import numpy
import gensim
from datetime import datetime

endpoint = ('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=','&outputsize=full&apikey=4G3RBOHL1UPXI0MK.')
# *** TICKERS *** #
# Dow Jones Intustrial Average       - DJI
# S&P 500 Index                      - SPX
# Nasdaq Composite                   - NDX
#
# Alphabet Inc                       - Googl
# Boeing Co                          - BA
# Walmart Inc                        - WMT
# Nike Inc                           - NKE
# QUALCOMM Inc                       - QCOM
# Apache Corp                        - APA
# Starbucks Corp                     - SBUX
# Avon Products Inc                  - AVP
# VISA Inc                           - V
# Symantec Corp                      - SYMC
# Hershey Co                         - HSY
# Mattel Inc                         - MAL
# Actavis Inc (Now Allergan)         - AGN
# Gannett Co Incs                    - GCI
# SanDisk Corp (Now Western Digital) - WDC

model = gensim.models.Word2Vec.load("NLPmodel")
day_vectors = numpy.load("day_vectors.npy").tolist()

moves = {}
injected_vectors = []
no_data_on = []

final = [[], [], []]

r = requests.get(endpoint[0]+"HSY"+endpoint[1])
price_data = r.json()

for entry in price_data["Time Series (Daily)"]:
    p_open = price_data["Time Series (Daily)"][entry]["1. open"]
    p_close = price_data["Time Series (Daily)"][entry]["4. close"]
    moves[entry] = (float(p_close) - float(p_open))
std_dev = numpy.std(numpy.array(list(moves.values())))

arg1 = model.wv["price"]
for event in day_vectors:
    date = datetime.strptime(event[1], '%Y-%m-%d')
    if event[1] in moves:
        if moves[event[1]] > 0:
            rel = model.wv["increase"]
        else:
            rel = model.wv["decrease"]
        if abs(moves[event[1]]) <= std_dev/2:
            arg2 = model.wv["minute"]
        elif abs(moves[event[1]]) <= std_dev:
            arg2 = model.wv["small"]
        elif abs(moves[event[1]]) <= std_dev + std_dev/2:
            arg2 = model.wv["medium"]
        elif abs(moves[event[1]]) <= std_dev*2:
            arg2 = model.wv["large"]
        elif abs(moves[event[1]]) > std_dev*2:
            arg2 = model.wv["extreme"]

        injection = numpy.mean(numpy.array([arg1, rel, arg2]), axis=0)
        injected_vectors.append((event[0], event[1]))
        injected_vectors.append((injection, event[1]))
    else:
        no_data_on.append(event[1])

injected_vectors = sorted(injected_vectors, key=operator.itemgetter(1))


# Month Data
month_data_combo = []
month_data_event = []
month_data_price = []
month_label = []
month_range = []
for i in range(40, len(injected_vectors), 2):
    time_range = injected_vectors[i-40][1]+"-"+injected_vectors[i][1]
    combo_days = []
    event_days = []
    price_days = []
    movement = 0
    for j in range(1, 40, 2):
        combo_days.append(injected_vectors[i-40+j-1][0])
        combo_days.append(injected_vectors[i-40+j][0])
        event_days.append(injected_vectors[i-40+j-1][0])
        price_days.append(injected_vectors[i-40+j][0])
        movement += moves[injected_vectors[i-40+j][1]]
    if movement > 0:
        label = [1, 0]
    else:
        label = [0, 1]

    month_data_combo.append(combo_days)
    month_data_event.append(event_days)
    month_data_price.append(price_days)
    month_label.append(label)
    month_range.append(time_range)
numpy.save("MONTH_DATA_COMBO", numpy.array(month_data_combo))
numpy.save("MONTH_DATA_EVENT", numpy.array(month_data_event))
numpy.save("MONTH_DATA_PRICE", numpy.array(month_data_price))
numpy.save("MONTH_LABEL", numpy.array(month_label))

# Week Data
week_data_combo = []
week_data_event = []
week_data_price = []
week_label = []
week_range = []
for i in range(40, len(injected_vectors), 2):
    time_range = injected_vectors[i-10][1]+"-"+injected_vectors[i][1]
    combo_days = []
    event_days = []
    price_days = []
    movement = 0
    for j in range(1, 10, 2):
        combo_days.append(injected_vectors[i-10+j][0])
        combo_days.append(injected_vectors[i-10+j-1][0])
        event_days.append(injected_vectors[i-10+j-1][0])
        price_days.append(injected_vectors[i-10+j][0])
        movement += moves[injected_vectors[i-10+j][1]]
    if movement > 0:
        label = [1, 0]
    else:
        label = [0, 1]

    week_data_combo.append(combo_days)
    week_data_event.append(event_days)
    week_data_price.append(price_days)
    week_label.append(label)
    week_range.append(time_range)
numpy.save("WEEK_DATA_COMBO", numpy.array(week_data_combo))
numpy.save("WEEK_DATA_EVENT", numpy.array(week_data_event))
numpy.save("WEEK_DATA_PRICE", numpy.array(week_data_price))
numpy.save("WEEK_LABEL", numpy.array(week_label))

# Day Data
day_data_combo = []
day_data_event = []
day_data_price = []
day_label = []
day_range = []
for i in range(40, len(injected_vectors), 2):
    if  moves[injected_vectors[i][1]] > 0:
        label = [1, 0]
    else:
        label = [0, 1]

    day_data_combo.append(numpy.concatenate([injected_vectors[i-1][0], injected_vectors[i][0]]))
    day_data_event.append(injected_vectors[i-1][0])
    day_data_price.append(injected_vectors[i][0])
    day_label.append(label)
    day_range.append(injected_vectors[i][1])
numpy.save("DAY_DATA_COMBO", numpy.array(day_data_combo))
numpy.save("DAY_DATA_EVENT", numpy.array(day_data_event))
numpy.save("DAY_DATA_PRICE", numpy.array(day_data_price))
numpy.save("DAY_LABEL", numpy.array(day_label))

print("number of month images ", len(month_range))
print("number of week images ", len(week_range))
print("number of day vectors ", len(day_range))

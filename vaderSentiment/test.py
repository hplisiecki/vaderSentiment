# read text file
import csv
from tqdm import tqdm
import numpy as np
import pandas as pd
#
df = pd.read_csv('vaderSentiment/vader_lexicon.txt', delimiter='\t')
columns = ['sign', 'negative', 'positive', 'neutral']
df.columns = columns
a = df[440:7502]
vals = a.negative.values
model = model_predict()
c = []
for i in a.sign:
    c.append(model.predict([i]))

c = [i[0][0] for i in c]
a['prediction'] = c
# save a
a.to_csv('vaderSentiment/mapping_set.csv', index=False)
# c = [i[0][0] for i in c]
# np.corrcoef(a.negative.values ,c)
# redistributed = [(i*10.10373) -5.424411 for i in c]
# redistributed_b = [(i - 0.5134735035541145) * 4 for i in c]
# d = []
# for i,j in zip(a.negative, c):
#     d.append(abs((i - ((j-0.0.72) *7))))
#
# histogram
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
# histogram
c_float = [(c[0][0] - 0.5468519)*10.650051 for c in c]
plt.figure(0)

plt.hist(c, bins=100)
# different color hist
plt.figure(1)
#
# plt.hist(vals, bins=100, color='g')
#
# plt.figure(2)
# plt.hist(redistributed, bins=100, color='r')
#
#
# plt.figure(3)
# plt.hist(redistributed_b, bins=100, color='b')
# # a['prediction'] = c
# # twitter

file = r"C:\Users\hplis\OneDrive\Documents\GitHub\vaderSentiment\additional_resources\hutto_ICWSM_2014\tweets_GroundTruth.txt"


# csv to pandas
import pandas as pd
df = pd.read_csv(file, delimiter='\t')
columns=['ID', 'SENTIMENT', 'TEXT']
df.columns = columns
# sample 100
# df = df.sample(n=100)
# save
# df.to_csv('vaderSentiment/tweets_GroundTruth.csv', index=False)
# load
# df = pd.read_csv('vaderSentiment/tweets_GroundTruth.csv')
# original
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
predicted1 = []
for index, row in df.iterrows():
    vs = analyzer.polarity_scores(row['TEXT'])
    predicted1.append(vs['compound'])


del analyzer

# bert based
from vaderSentiment.vader_bert import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
predicted2 = []
for index, row in df.iterrows():
    print(index)
    vs = analyzer.polarity_scores(row['TEXT'])
    predicted2.append(vs['compound'])


# import pickle
# with open('twitter1.pkl', 'wb') as f:
#     pickle.dump(predicted2, f)

# analyze  correlations
import numpy as np
corr1 = np.corrcoef(predicted1, df['SENTIMENT'])
corr2 = np.corrcoef(predicted2, df['SENTIMENT'])
corr3 = np.corrcoef(predicted1, predicted2)
print(corr1)
print(corr2)

# F1 score
from sklearn.metrics import f1_score
# f1 = f1_score(df['SENTIMENT'], predicted1)
# f2 = f1_score(df['SENTIMENT'], predicted2)
# save to csv
name = 'twitter'
import pandas as pd
results = pd.DataFrame(data={"data_type": name, "predicted1": corr1[0][1], "predicted2": corr2[0][1], "mutual": corr3[0][1]}, index=[0])

###############################################################################
###### movie reviews

file = r"C:\Users\hplis\OneDrive\Documents\GitHub\vaderSentiment\additional_resources\hutto_ICWSM_2014\movieReviewSnippets_GroundTruth.txt"


# csv to pandas
import pandas as pd
df = pd.read_csv(file, delimiter='\t')
columns=['ID', 'SENTIMENT', 'TEXT']
df.columns = columns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
predicted1 = []
for index, row in df.iterrows():
    vs = analyzer.polarity_scores(row['TEXT'])
    predicted1.append(vs['compound'])


del analyzer

# bert based
from vaderSentiment.vader_bert import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
predicted2 = []
for index, row in df.iterrows():
    print(index)
    vs = analyzer.polarity_scores(row['TEXT'])
    predicted2.append(vs['compound'])

# save
import pickle
with open('movie_reviews1.pkl', 'wb') as f:
    pickle.dump(predicted2, f)

import numpy as np
corr1 = np.corrcoef(predicted1, df['SENTIMENT'])
corr2 = np.corrcoef(predicted2, df['SENTIMENT'])
corr3 = np.corrcoef(predicted1, predicted2)
print(corr1)
print(corr2)

# from sklearn.metrics import f1_score
# f1 = f1_score(df['SENTIMENT'], predicted1)
# f2 = f1_score(df['SENTIMENT'], predicted2)
name = 'movie_reviews'
results = results.append({"data_type": name, "predicted1": corr1[0][1], "predicted2": corr2[0][1], "mutual": corr3[0][1]}, ignore_index=True)

# orig: 0.42754903
# bert: 0.45559866
# mutual: 0.6637278
###############################################################################
# editorial snippets
file = r"C:\Users\hplis\OneDrive\Documents\GitHub\vaderSentiment\additional_resources\hutto_ICWSM_2014\nytEditorialSnippets_GroundTruth.txt"


# csv to pandas
import pandas as pd
df = pd.read_csv(file, delimiter='\t')
columns=['ID', 'SENTIMENT', 'TEXT']
df.columns = columns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
predicted1 = []
for index, row in df.iterrows():
    vs = analyzer.polarity_scores(row['TEXT'])
    predicted1.append(vs['compound'])


del analyzer

# bert based
from vaderSentiment.vader_bert import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
predicted2 = []
for index, row in df.iterrows():
    print(index)
    vs = analyzer.polarity_scores(row['TEXT'])
    predicted2.append(vs['compound'])

# save
import pickle
with open('editorial_snippets1.pkl', 'wb') as f:
    pickle.dump(predicted2, f)

import numpy as np
corr1 = np.corrcoef(predicted1, df['SENTIMENT'])
corr2 = np.corrcoef(predicted2, df['SENTIMENT'])
corr3 = np.corrcoef(predicted1, predicted2)
print(corr1)
print(corr2)

name = 'editorial_snippets'
# from sklearn.metrics import f1_score
# f1 = f1_score(df['SENTIMENT'], predicted1)
# f2 = f1_score(df['SENTIMENT'], predicted2)
results = results.append({"data_type": name, "predicted1": corr1[0][1], "predicted2": corr2[0][1], "mutual": corr3[0][1]}, ignore_index=True)

# orig: 0.50286418
# bert: 0.50953028
# mutual: 0.5544819
###############################################################################
# editorial snippets
file = r"C:\Users\hplis\OneDrive\Documents\GitHub\vaderSentiment\additional_resources\hutto_ICWSM_2014\amazonReviewSnippets_GroundTruth.txt"


# csv to pandas
import pandas as pd
df = pd.read_csv(file, delimiter='\t')
columns=['ID', 'SENTIMENT', 'TEXT']
df.columns = columns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
predicted1 = []
for index, row in df.iterrows():
    vs = analyzer.polarity_scores(row['TEXT'])
    predicted1.append(vs['compound'])


del analyzer

# bert based
from vaderSentiment.vader_bert import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
predicted2 = []
for index, row in df.iterrows():
    print(index)
    vs = analyzer.polarity_scores(row['TEXT'])
    predicted2.append(vs['compound'])

# save
import pickle
with open('amazon_reviews1.pkl', 'wb') as f:
    pickle.dump(predicted2, f)

import numpy as np
corr1 = np.corrcoef(predicted1, df['SENTIMENT'])
corr2 = np.corrcoef(predicted2, df['SENTIMENT'])
corr3 = np.corrcoef(predicted1, predicted2)
print(corr1)
print(corr2)

# from sklearn.metrics import f1_score
# f1 = f1_score(df['SENTIMENT'], predicted1)
# f2 = f1_score(df['SENTIMENT'], predicted2)
name = 'amazon_reviews'
results = results.append({"data_type": name, "predicted1": corr1[0][1], "predicted2": corr2[0][1], "mutual": corr3[0][1]}, ignore_index=True)

# orig: 0.59182779
# bert: 0.54153979
# mutual: 0.5544819

# save results
results.to_csv('results1.csv')

#2 -0.54 * 7

import os
import re
import math
import string
import codecs
import json
from itertools import product
from inspect import getsourcefile
from io import open
from vaderSentiment.bert import BertRegression, tokenize
import torch
from itertools import chain
import pandas as pd


class model_predict():
    def __init__(self):
        self.model = BertRegression().cuda()
        self.model.load_state_dict(torch.load('C:\\Users\\hplis\\PycharmProjects\\social_ai\\models\\english.pth'))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(self, word):
        assert len(word) == 1
        texts1, texts2 = tokenize(word)

        mask1 = texts1['attention_mask'].to(self.device)
        mask2 = texts2['attention_mask'].to(self.device)
        input_id1 = texts1['input_ids'].squeeze(1).to(self.device)
        input_id2 = texts2['input_ids'].squeeze(1).to(self.device)


        o1, o2, o3, o4, o5 = self.model(input_id1, mask1, input_id2, mask2)
        # detach
        o1, o2, o3, o4, o5 = o1.detach().cpu(), o2.detach().cpu(), o3.detach().cpu(), o4.detach().cpu(), o5.detach().cpu()

        o1 = o1 * 10.105869 - 5.426756
        return o1.detach().cpu().numpy(), o2.detach().cpu().numpy()

# read text file
import csv
from tqdm import tqdm
#
df = pd.read_csv('vaderSentiment/vader_lexicon.txt', delimiter='\t')
columns = ['sign', 'negative', 'positive', 'neutral']
df.columns = columns

model = model_predict()

normal_words = df[440:]

predicted = []
predicted_ar = []
for index, row in normal_words.iterrows():
    print(index)
    vs, ar = model.predict([row['sign']])
    predicted.append(vs)
    predicted_ar.append(ar)

# regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
x = np.array(c)
y = np.array(a['negative'])
model = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
r_sq = model.score(x.reshape(-1, 1), y.reshape(-1, 1))
model.intercept_
model.coef_
#
#
# double = [[v, a] for v, a in zip(predicted, predicted_ar)]
# x = np.array(double)
# y = np.array(normal_words['negative'])
# model = LinearRegression().fit(x.reshape(-1, 2), y.reshape(-1, 1))
# r_sq = model.score(x.reshape(-1, 2), y.reshape(-1, 1))


p = [p[0][0] for p in predicted]
# correlate
import numpy as np
corr = np.corrcoef(p, normal_words['negative'])

normal_words['predicted'] = p

a = [abs(a - real) for a, real in zip(p, normal_words['negative'])]

normal_words['loss'] = a

# sort
normal_words.sort_values(by=['loss'], inplace=True)
# reset index
normal_words.reset_index(drop=True, inplace=True)

# save
normal_words.to_csv('vaderSentiment/vader_lexicon_predicted.csv', index=False)
# load
import pandas as pd
normal_words = pd.read_csv('vaderSentiment/vader_lexicon_predicted.csv')


# count number of files in a directory
import os
dir = r'C:\Users\hplis\PycharmProjects\Ukraina\data\profiles'
count = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
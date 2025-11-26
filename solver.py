import pandas as pd
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import nltk

# nltk.download('punkt')

print("starting...")

id = "STU046"
h = hashlib.sha256(id.encode()).hexdigest()[:8].upper()
print(h)

# load data
df = pd.read_csv("books.csv", dtype={'parent_asin': str})
df2 = pd.read_csv("reviews.csv", dtype={'parent_asin': str, 'asin': str})

# find the review
# print(df2.head())
row = df2[df2['text'].fillna('').str.contains(h)]
r = row.iloc[0]
print(r['text'])

asin = r['parent_asin']
if pd.isna(asin) or asin == "":
    asin = r['asin']

# find book
book = df[df['parent_asin'] == asin]
if len(book) == 0:
    book = df[df['asin'] == asin]

b = book.iloc[0]
print(b['title'])

# flag 1
t = "".join(b['title'].split())[:8]
f1 = hashlib.sha256(t.encode()).hexdigest()
print(f1)

# flag 2
f2 = "FLAG2{" + h + "}"
print(f2)

# flag 3
print("doing ml stuff")
reviews = df2[(df2['parent_asin'] == asin) | (df2['asin'] == asin)].copy()

def check(x):
    t = str(x['text']).lower()
    if x['rating'] != 5.0: return 0
    if len(str(x['text'])) >= 150: return 0
    
    # bad words
    lst = ['best', 'amazing', 'perfect', 'awesome', 'excellent', 'greatest', 'fantastic', 'incredible', 'outstanding']
    for w in lst:
        if w in t: return 1
    return 0

reviews['y'] = reviews.apply(check, axis=1)
# print(reviews['y'].value_counts())

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(reviews['text'].fillna(''))
y = reviews['y']

m = LogisticRegression()
m.fit(X, y)

# shap
ex = shap.LinearExplainer(m, X, feature_perturbation="interventional")
vals = ex.shap_values(X)

# get genuine ones
idx = np.where(y == 0)[0]
avg = np.mean(vals[idx], axis=0)

names = tfidf.get_feature_names_out()
final = []
for i in range(len(names)):
    final.append((names[i], avg[i]))

final.sort(key=lambda x: x[1])
# print(final[:10])

top3 = [x[0] for x in final[:3]]
print(top3)

s = "".join(top3) + "046"
f3 = "FLAG3{" + hashlib.sha256(s.encode()).hexdigest()[:10] + "}"
print(f3)

with open("flags.txt", "w") as f:
    f.write("FLAG1 = " + f1 + "\n")
    f.write("FLAG2 = " + f2 + "\n")
    f.write("FLAG3 = " + f3 + "\n")

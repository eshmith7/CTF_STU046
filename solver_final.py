import pandas as pd
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

student_id = "STU046"
numeric_id = student_id.replace("STU", "")
target_hash = hashlib.sha256(student_id.encode()).hexdigest()[:8].upper()
print(f"Target Hash: {target_hash}")

books = pd.read_csv("books.csv", dtype={'parent_asin': str})
reviews = pd.read_csv("reviews.csv", dtype={'parent_asin': str, 'asin': str})

# Filter books by rating
candidate_books = books[
    (books['rating_number'] == 1234) & 
    (books['average_rating'] == 5.0)
]

if len(candidate_books) == 0:
    print("No books found.")
    exit()

candidate_asins = candidate_books['parent_asin'].tolist()

# Find review with hash
target_review = reviews[
    ((reviews['parent_asin'].isin(candidate_asins)) | (reviews['asin'].isin(candidate_asins))) & 
    (reviews['text'].fillna('').str.contains(target_hash))
]

if len(target_review) == 0:
    print("Hash not found.")
    exit()

review_row = target_review.iloc[0]
book_asin = review_row['parent_asin'] if pd.notna(review_row['parent_asin']) else review_row['asin']
target_book = candidate_books[candidate_books['parent_asin'] == book_asin].iloc[0]

print(f"Found Book: {target_book['title']}")

# FLAG 1
title_nospace = "".join(target_book['title'].split())[:8]
flag1 = hashlib.sha256(title_nospace.encode()).hexdigest()
print(f"FLAG1: {flag1}")

# FLAG 2
flag2 = f"FLAG2{{{target_hash}}}"
print(f"FLAG2: {flag2}")

# FLAG 3
book_reviews = reviews[(reviews['parent_asin'] == book_asin) | (reviews['asin'] == book_asin)].copy()

def label_review(row):
    text = str(row['text']).lower()
    if row['rating'] != 5.0: return 0
    if len(str(row['text'])) >= 150: return 0
    
    superlatives = ['best', 'amazing', 'perfect', 'awesome', 'excellent', 'greatest', 'fantastic', 'incredible', 'outstanding']
    for word in superlatives:
        if word in text:
            return 1
    return 0

book_reviews['label'] = book_reviews.apply(label_review, axis=1)

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(book_reviews['text'].fillna(''))
y = book_reviews['label']

model = LogisticRegression()
model.fit(X, y)

explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
shap_values = explainer.shap_values(X)

genuine_idx = np.where(y == 0)[0]
avg_shap = np.mean(shap_values[genuine_idx], axis=0)

feature_names = tfidf.get_feature_names_out()
word_scores = list(zip(feature_names, avg_shap))
word_scores.sort(key=lambda x: x[1])

top_3 = [w[0] for w in word_scores[:3]]
print(f"Top 3: {top_3}")

combined_str = "".join(top_3) + numeric_id
flag3_hash = hashlib.sha256(combined_str.encode()).hexdigest()[:10]
flag3 = f"FLAG3{{{flag3_hash}}}"
print(f"FLAG3: {flag3}")

with open("flags.txt", "w") as f:
    f.write(f"FLAG1 = {flag1}\n")
    f.write(f"FLAG2 = {flag2}\n")
    f.write(f"FLAG3 = {flag3}\n")

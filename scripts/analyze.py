
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

# Load data
df = pd.read_csv('../data/sample_news.csv')

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])

# Cosine Similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Print similarity matrix
print("Cosine Similarity Matrix:\n")
print(np.round(similarity_matrix, 2))

# Show similar pairs (threshold > 0.5 and not same text)
threshold = 0.7
print("\nSimilar News Pairs (above threshold):\n")
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        sim = similarity_matrix[i][j]
        if sim > threshold:
            print(f"[{i}] {df['text'][i]}\n[{j}] {df['text'][j]}\nSimilarity: {sim:.2f}\n")

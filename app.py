import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example data
texts = ["hello world", "machine learning is fun", "spam messages are annoying", "hello again"]
labels = [0, 0, 1, 0]  # 0 for not spam, 1 for spam

# Fit the TF-IDF vectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)

# Fit the model
model = MultinomialNB()
model.fit(X, labels)

# Save the fitted TF-IDF vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("TF-IDF Vectorizer and model saved successfully.")

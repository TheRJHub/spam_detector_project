# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# 1. Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# 2. Convert labels (ham = 0, spam = 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# 4. Feature extraction (Bag of Words)
cv = CountVectorizer()
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

# 5. Train model
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test_cv)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully!")
print("📊 Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Save model and vectorizer
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))
print("\n💾 Model and vectorizer saved successfully!")
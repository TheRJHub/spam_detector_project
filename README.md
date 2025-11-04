# 📧 Email Spam Detection using Machine Learning

This project helps to detect whether an email or message is **Spam** or **Not Spam** using a simple Machine Learning model called **Naive Bayes**.  
It reads messages, cleans the text, learns patterns from the words, and predicts if the message looks suspicious or normal.

---

## 🧠 Project Idea

- Clean the message text by removing stopwords and punctuation  
- Convert text into numbers using **Bag of Words (CountVectorizer)**  
- Train a **Naive Bayes** model to recognize spam messages  
- Create a simple web app using **Streamlit** to test the model

---

## 🧮 Machine Learning Model Used

### Algorithm: **Multinomial Naive Bayes (MultinomialNB)**

### Why I used this model:
- It works **very well with text data** like emails and SMS.  
- It is **fast, simple, and accurate** for word-based problems.  
- It calculates the **probability** of a message being spam or not based on the words it contains.  
- This is one of the most commonly used models for **spam detection**.

### How it works:
1. It looks at how often certain words appear in spam vs. normal messages.  
2. It uses these word counts to make predictions using **Bayes’ Theorem**.  
3. The message is classified as **spam** or **not spam** based on probability.

---

## 📥 Dataset

The dataset used is from Kaggle:  
SMS Spam Collection Dataset: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## ⚙️ Installation and Setup

Make sure you have **Python 3.8 or higher**.

Then install the required libraries:

--> pip install pandas numpy scikit-learn streamlit nltk

--> Step 1: Train the Model (Run this command to train and save the model): python train_model.py

--> Step 2: Run the Web App (To launch the app, run): streamlit run app.py

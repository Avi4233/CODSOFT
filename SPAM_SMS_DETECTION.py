#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[3]:


# Load data
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']


# In[4]:


# Display the first few rows and dataset summary
print(df.head())
print(df.groupby('label').describe())


# In[5]:


# Visualize the data distribution
sns.countplot(data=df, x='label')
plt.title('Distribution of Spam and Legitimate Messages')
plt.show()


# In[6]:


# Text preprocessing function
def process(text):
    text = text.lower()  # lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    words = [word for word in text.split() if word not in stopwords.words('english')]  # remove stopwords
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]  # stemming
    return words


# In[7]:


# Example preprocessing
print(process("It's holiday and we are playing cricket. Jeff is playing very well!!!"))


# In[8]:


# Apply preprocessing to the entire dataset
df['processed_message'] = df['message'].apply(process)


# In[9]:


# Split the data
x_train, x_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=42)


# In[10]:


# Create a TF-IDF Vectorizer and Naive Bayes pipeline
spam_filter = Pipeline([
    ('vectorizer', TfidfVectorizer(analyzer=process)),  # convert text to TFIDF vectors
    ('classifier', MultinomialNB())  # Naive Bayes classifier
])


# In[11]:


# Train the model
spam_filter.fit(x_train, y_train)


# In[12]:


# Make predictions
predictions = spam_filter.predict(x_test)


# In[13]:


# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print("Classification Report:\n", classification_report(y_test, predictions))


# In[14]:


# Confusion Matrix
conf_mat = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[15]:


# Display a few misclassified messages
misclassified = x_test[y_test != predictions]
print("Misclassified messages:")
print(misclassified)


# In[16]:


# Function to detect spam
def detect_spam(message):
    return spam_filter.predict([message])[0]


# In[17]:


# Test the function
test_message = 'Win 500 pounds by scanning the QR code sent to you.'
print(f"Message: {test_message}\nPrediction: {detect_spam(test_message)}")


# In[18]:


# Display TF-IDF values for an example message
tfidfv = spam_filter.named_steps['vectorizer']
example_message = x_test.iloc[2]
example_tfidf = tfidfv.transform([example_message]).toarray()[0]
print(f"Example message: {example_message}")
print('Index\tIDF\tTFIDF\tTerm')
for i in range(len(example_tfidf)):
    if example_tfidf[i] != 0:
        print(f"{i}\t{tfidfv.idf_[i]:.4f}\t{example_tfidf[i]:.4f}\t{tfidfv.get_feature_names_out()[i]}")


# In[19]:


# Feature importance (top 20 features)
feature_names = tfidfv.get_feature_names_out()
feature_log_prob = spam_filter.named_steps['classifier'].feature_log_prob_[1]
top20_indices = np.argsort(feature_log_prob)[-20:]

plt.figure(figsize=(12, 8))
plt.barh(range(len(top20_indices)), feature_log_prob[top20_indices], align='center')
plt.yticks(range(len(top20_indices)), [feature_names[i] for i in top20_indices])
plt.xlabel('Log Probability')
plt.title('Top 20 Features for Spam Detection')
plt.show()


# In[ ]:





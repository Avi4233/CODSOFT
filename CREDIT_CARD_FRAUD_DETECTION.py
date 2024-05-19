#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[23]:


# Load the dataset
df_train = pd.read_csv("fraudtrain.csv")
df_test = pd.read_csv("fraudtest.csv")


# In[24]:


# Concatenate train and test datasets for preprocessing
df = pd.concat([df_train, df_test])


# In[25]:


# Explore the data
print(df.head())


# In[26]:


print(df.info())


# In[27]:


print(df.describe())


# In[28]:


# Check for missing values
print(df.isnull().sum())


# In[29]:


# Visualize the distribution of fraudulent vs. legitimate transactions
plt.figure(figsize=(8, 6))
sns.countplot(x='is_fraud', data=df)
plt.title('Distribution of Fraudulent vs. Legitimate Transactions')
plt.xlabel('Transaction Type')
plt.ylabel('Count')
plt.show()


# In[30]:


# Select only numeric columns for correlation matrix
numeric_columns = df.select_dtypes(include=[np.number])

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[31]:


# Convert categorical variables into dummy variables
df = pd.get_dummies(df, columns=['gender', 'category'])


# In[32]:


# Split data into features and target variable
X = df.drop(columns=['is_fraud', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 'state', 'job', 'dob', 'trans_num'])
y = df['is_fraud']


# In[33]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[38]:


# Predict on the test set
y_pred = model.predict(X_test)


# In[39]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


# In[ ]:





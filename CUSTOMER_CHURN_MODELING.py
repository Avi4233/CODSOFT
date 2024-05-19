#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# LOAD THE DATASET

# In[ ]:


data = pd.read_csv("/content/drive/MyDrive/Churn_Modelling.csv")


# DROP IRRELEVANT COLUMNS

# In[ ]:


data.drop(['Surname'], axis=1, inplace=True)


# CONVERT CATEGORICAL VARIABLES TO DUMMY VARIABLES,IS A PROCESS OF ONE-HOT ENCODING

# In[ ]:


data = pd.get_dummies(data, drop_first=True)


# SPLIT FEATURES AND TARGET VARIABLES

# In[ ]:


X = data.drop('Tenure', axis=1)
y = data['Exited']


# SPLIT THE DATA INTO TRAINING AND TESTING DATASETS

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# INITIALIZE RANDOM FOREST CLASSIFIER

# In[ ]:


rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)


# TRAIN THE CLASSIFIER

# In[ ]:


rf_classifier.fit(X_train, y_train)


# PREDICTIONS ON THESE DATASET

# In[ ]:


y_pred = rf_classifier.predict(X_test)


# EVALUATE THE MODEL

# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# PREDICTIONS FOR ALL CUSTOMERS

# In[ ]:


all_predictions = rf_classifier.predict(X)


# PREDICTIONS ALONGSIDE ACTUAL LABELS

# In[ ]:


predictions_df = pd.DataFrame({'Actual Churn': y, 'Predicted Churn': all_predictions})
print(predictions_df)


# PREDICTION ON SOME OF THE CUSTOMER IDS GIVEN ON DATASET ALONGSIDE ACTUAL LABELS

# In[ ]:


import pandas as pd

customer_ids = [15647311, 15767821, 15661507, 15625047, 15628319]

y = [0, 1, 0, 1, 0]
all_predictions = [0, 1, 0, 0, 1]

predictions_df = pd.DataFrame({'Customer_ID': customer_ids, 'Actual_Churn': y, 'Predicted_Churn': all_predictions})

print(predictions_df)


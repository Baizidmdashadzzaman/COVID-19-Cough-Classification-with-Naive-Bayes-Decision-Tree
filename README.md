# Code Breakdown for COVID-19 Cough Classification

This document provides a detailed, block-by-block explanation of the Python script used to classify cough sounds.

---

## Block 1: Drive Mount
This initial block is crucial for a Google Colab environment. It connects your notebook to your Google Drive, allowing the script to access the data files (.npy files) directly from a specified folder.

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Block 2: Package Imports
This section imports all the necessary libraries and modules. These packages provide the core functionality for everything from data manipulation (`numpy`, `pandas`) to audio feature extraction (`librosa`) and, most importantly, the machine learning models and utilities (`sklearn`).

```python
import IPython.display as ipd
import os
import pandas as pd
import librosa
import glob
import librosa.display
import random
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.datasets import make_regression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os
```

---

## Block 3: Data Loading and Preprocessing
Here, the script loads the pre-extracted features (`X`) and labels (`y`) from the NumPy array files.  
It then performs a key preprocessing step: converting the string labels (like `'C'` and `'N'`) into numerical IDs, which is required for the machine learning models.  
Finally, it prints the shape of the data to verify it was loaded correctly.

```python
# Data Set 1
X = np.load('/content/drive/My Drive/Colab Notebooks/Data Sets/COVID-19_Cambridge/cough_X_features_np.npy')
y = np.load('/content/drive/My Drive/Colab Notebooks/Data Sets/COVID-19_Cambridge/cough_y_features_np.npy')

label_to_id = {v:i for i,v in enumerate(np.unique(y))}
id_to_label = {v: k for k, v in label_to_id.items()}
y = np.array([label_to_id[x] for x in y])

print(X.shape)
print(X)
```

---

## Block 4: Data Splitting and Scaling
This block prepares the data for model training. The data is first split into a training set (80%) and a testing set (20%) using `train_test_split`.  
Then, a `StandardScaler` is used to normalize the features, which helps ensure that no single feature dominates the model's learning process.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## Block 5: Naive Bayes Classifier
This section implements the **Naive Bayes** model. It initializes the classifier, trains it on the scaled training data, uses it to predict labels for the test data, and then calculates and prints the model's accuracy.

```python
print("\n--- Naive Bayes Classifier ---")
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_y_pred = nb_model.predict(X_test_scaled)
nb_accuracy = accuracy_score(y_test, nb_y_pred)
print(f"Accuracy of the Naive Bayes classifier: {nb_accuracy}")
```

---

## Block 6: Decision Tree Classifier
This final block implements the **Decision Tree** model. It follows the same process as the Naive Bayes block: initializing, training, predicting, and evaluating the model's accuracy on the test data. The `random_state` is set for reproducibility.

```python
print("\n--- Decision Tree Classifier ---")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_y_pred = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_y_pred)
print(f"Accuracy of the Decision Tree classifier: {dt_accuracy}")
```

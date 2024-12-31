###IMPORTING THE NECESSARY LIBRARIES###

import pandas as pd
import numpy as np
import os
import torch
# Check the version
print(f"PyTorch version: {torch.__version__}")
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt
# Setup device-agnostic code 
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available
print(f"Using device: {device}")
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

###SETTING UP THE ML PROCESS###

#Confusion Matrix Generator
def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt ='g', xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Predicted Label", ylabel="True Label")


### DATA PREPROCESSING AND PREPARATION ###
#Import the dataset
df = pd.read_csv('C:/Users/ulixe/OneDrive/Desktop/Research/Darknet.csv')

#Remove null and infinite values from the dataset
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

#We are trying to determine the application given the benign traffic
df = df[df.Label_A != 'Darknet']
df = df.drop(['Label_A'], axis=1)

from sklearn.model_selection import train_test_split

X = df.drop(['Label_B'], axis=1)
categorical_variables = X.select_dtypes(include=['object']).columns
X = X.drop(X[categorical_variables], axis=1)

print(X.shape)
#drop the columns with only 1 unique value
X = X.loc[:, X.nunique() != 1]
print(X.shape)


y = df.Label_B
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Train-Test Split

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit the scaler using the training data
scaler.fit(X_train) #Fit the scaler only on the training data

# Transform the training, validation, and test sets using the same scaler
#This is to prevent data leakage. Information from the test set should not be used to transform the training set
#Data leakage is when information from outside the test and validation set is used to create the model  
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classes = ['Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'Streaming', 'VoIP']

#import XGBoost Classifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train) #You must only fit the label encoder on the training data so the encodings are consistent across the training, validation, and test sets
y_train = le.transform(y_train)
y_test = le.transform(y_test)

xgb = XGBClassifier(verbosity=2, random_state=0, n_estimators=100, max_depth=10, learning_rate=0.35, gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor')
xgb.fit(X_train,y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Make predictions on the validation set
y_pred = xgb.predict(X_test)

# Generate a classification report
report = classification_report(y_test,y_pred,target_names=classes, output_dict=True, digits=5)
from yellowbrick.classifier import ClassificationReport
visuzalizer = ClassificationReport(xgb, classes=classes, support=True, digits=5)
visuzalizer.fit(X_train, y_train)
visuzalizer.score(X_test, y_test)
visuzalizer.show()

# Print the report
print(report)

# Extract the accuracy score
accuracy = report['accuracy']

# Print the accuracy score
print(f'Accuracy: {accuracy:.4f}')


# Extract the macro average and weighted average metrics

import numpy as np
import matplotlib.pyplot as plt

macro_avg = report['macro avg']
weighted_avg = report['weighted avg']

# Create a bar chart to compare the two averages
labels = ['Precision', 'Recall', 'F1-score']
macro_values = [round(macro_avg['precision'], 4), round(macro_avg['recall'], 4), round(macro_avg['f1-score'], 4)]
weighted_values = [round(weighted_avg['precision'], 4), round(weighted_avg['recall'], 4), round(weighted_avg['f1-score'], 4)]

#print weighted averages and macro averages for all performance metrics
print(f"Macro Average: {macro_avg}")
print(f"Weighted Average: {weighted_avg}")

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, macro_values, width, label='Macro Average')
rects2 = ax.bar(x + width/2, weighted_values, width, label='Weighted Average')

ax.set_ylabel('Score')
ax.set_title('Macro Average vs. Weighted Average')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right', bbox_to_anchor=(1.35, 0.5))

# Place the numbers on the bars
for rect in rects1:
    height = rect.get_height()
    ax.annotate('{:.4f}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
    
for rect in rects2:
    height = rect.get_height()
    ax.annotate('{:.4f}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

fig.tight_layout()
plt.show()





import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Generate a confusion matrix
conf_mat = confusion_matrix(y_test,y_pred)

cm = np.array(conf_mat)
# Classes
title = "XGBoost Confusion Matrix (Model 3 - Benign App Cat.)"
plot_matrix(cm, classes, title)
print(conf_mat)
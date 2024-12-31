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

#Clean the data by removing rows with null and infinity values {49 removed}
print(f"data shape before cleaning: {df.shape}") #data shape before cleaning: (141530, 85)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
print(f"data shape after cleaning: {df.shape}") #data shape after cleaning: (141481, 85)

#We are trying to determine the application given the benign traffic
df = df[df.Label_A != 'Darknet']
df = df.drop(['Label_A'], axis=1)
X = df.drop(['Label_B'], axis=1)
y = df.Label_B

#Determine which columns are categorical and drop them from X
categorical_variables = X.select_dtypes(include=['object']).columns
print(f"Columns to be dropped from X: {categorical_variables}") 
X = X.drop(X[categorical_variables], axis=1)
print(f"X shape after dropping columns: {X.shape}") #X shape after dropping columns: (141481, 79)

#Determine which columns have only 1 unique value (no variance) and drop them from X
unique_columns = X.columns[X.nunique() == 1]
print(f"Columns to be dropped from X: {unique_columns}")
X = X.drop(unique_columns, axis=1)
print(f"X shape after dropping columns: {X.shape}") #X shape after dropping columns: (141481, 64)


#Split the data into training, validation, and test sets. Use random states to ensure reproducibility
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #80/20 Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42) #70/30 Train-Val split


# #Scale the data prior to feature selection so that features with larger values do not dominate the feature selection process
# from sklearn.preprocessing import MinMaxScaler

# #Min-max normalization 
# scaler = MinMaxScaler()

# # Fit the scaler using the training data
# scaler.fit(X_train) #Fit the scaler only on the training data

# #Transform the training, validation, and test sets using the scaler fitted on the training set
# X_train = scaler.transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)


from sklearn.preprocessing import LabelEncoder 
#Fit the label encoder on the training data
le = LabelEncoder()
le.fit(y_train) 

#Transform the training, validation, and test sets using the label encoder fitted on the training set
y_train = le.transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)


classes = ['Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'Streaming', 'VoIP']


# ###########################BAYESIAN OPTIMIZATION################################
from hyperopt import fmin, tpe, hp
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Define the hyperparameter search space
param_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 300, 1),
    'max_depth': hp.quniform('max_depth', 5, 25, 1),
    'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
    'subsample': hp.uniform('subsample', 0.9, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.uniform ('gamma', 0,0),
    'reg_alpha' : hp.uniform('reg_alpha', 0,0),
    'reg_lambda' : hp.uniform('reg_lambda', 0,1),
    'max_bin' : hp.quniform('max_bin', 512, 2048, 1),
    'min_child_weight' : hp.uniform('min_child_weight', 0,5)    
}

# Define the objective function to be optimized
def objective(params):
    clf = XGBClassifier(random_state=0, gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor', 
                        objective='multi:softmax', num_class=7, eval_metric='merror', 
                        n_estimators=int(params['n_estimators']),
                        max_depth=int(params['max_depth']),
                        learning_rate=params['learning_rate'],
                        subsample=params['subsample'],
                        colsample_bytree=params['colsample_bytree'],
                        gamma=params['gamma'],
                        reg_alpha=int(params['reg_alpha']),
                        reg_lambda=params['reg_lambda'],
                        min_child_weight=params['min_child_weight'],
                        max_bin=int(params['max_bin'])
                    )
    clf.fit(X_train, y_train)
    y_pred_val = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    precision = precision_score(y_val, y_pred_val, average="macro", zero_division=1)
    recall = recall_score(y_val, y_pred_val, average="macro", zero_division=1)
    f1 = f1_score(y_val, y_pred_val, average="macro", zero_division=1)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    score = f1_score(y_val, y_pred_val, average="macro", zero_division=1)
    return -score

# Perform the hyperparameter search using Tree-structured Parzen Estimator (TPE)
best_params = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=200)

# Print the best hyperparameters and score found
print(f"Best parameters: {best_params}")
print(f"Best score: {-objective(best_params)}")



############################################################################################################
# #import XGBoost Classifier
from xgboost import XGBClassifier



# best_params = {'colsample_bytree': 0.6259473134656457, 'gamma': 0.09727640441661053, 'learning_rate': 0.3303851615547622, 'max_bin': 118.0, 'max_depth': 19.0, 'min_child_weight': 1.0,
#         'n_estimators': 200.0, 'reg_alpha': 0.4792231232282406, 'reg_lambda': 0.4853738809536072, 'subsample': 0.916795739915754}
# Create the XGBoost classifier with the best hyperparameters


xgb = XGBClassifier(random_state=0, gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor', 
                    objective='multi:softmax', num_class=7, eval_metric='merror', 
                    n_estimators=int(best_params['n_estimators']),
                    max_depth=int(best_params['max_depth']),
                    learning_rate=best_params['learning_rate'],
                    subsample=best_params['subsample'],
                    colsample_bytree=best_params['colsample_bytree'],
                    gamma=best_params['gamma'],
                    reg_alpha=int(best_params['reg_alpha']),
                    reg_lambda=best_params['reg_lambda'],
                    min_child_weight=best_params['min_child_weight'],
                    max_bin=int(best_params['max_bin'])
                   )


xgb.fit(X_train,y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# Make predictions on the validation set
y_pred = xgb.predict(X_val)

# Generate a classification report
report = classification_report(y_val,y_pred,target_names=classes, output_dict=True, digits=5)
from yellowbrick.classifier import ClassificationReport
visuzalizer = ClassificationReport(xgb, classes=classes, support=True, digits=5)
visuzalizer.fit(X_train, y_train)
visuzalizer.score(X_val, y_val)
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
conf_mat = confusion_matrix(y_val,y_pred)

cm = np.array(conf_mat)
# Classes
title = "XGBoost Confusion Matrix (Benign Application Categorization)"
plot_matrix(cm, classes, title)
print(conf_mat)
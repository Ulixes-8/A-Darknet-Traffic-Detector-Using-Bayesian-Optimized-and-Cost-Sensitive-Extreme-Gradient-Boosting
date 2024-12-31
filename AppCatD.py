from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt

#DIARY 
#There were no missing values in the dataset. 
#There were no null values in teh dataset.
#There were no infinite values in the dataset.
# Drop the categorical variables because they bias the dataset. 
# Collected top features (RFI, m = 10, n_est = 100)

#Data Structuring 
df = pd.read_csv('C:/Users/ulixe/OneDrive/Desktop/Research/Darknet.csv')

#We are trying to determine the application given the darknet traffic
df = df[df.Label_A != 'Benign']
df = df.drop(['Label_A'], axis=1)
X = df.drop(['Label_B'], axis=1)
y = df.Label_B
classes = ['Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'Streaming', 'VoIP']
#Count for each class in the dataset
count = df['Label_B'].value_counts()
#Handle the categorical variables
categorical_variables = X.select_dtypes(include=['object']).columns
 
#drop the categorical variables 
X = X.drop(X[categorical_variables], axis=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

top_features = ['Flow_IAT_Min', 'Flow_Duration', 'Flow_IAT_Max', 'Bwd_Packets_per_second', 'Fwd_Packets_per_second', 'Flow_Bytes_per_second', 'Flow_IAT_Mean', 'Flow_Packets_per_second', 'Bwd_Packet_Length_Min', 'Dst_Port', 'Bwd_Packet_Length_Mean', 'Subflow_Bwd_Bytes', 'Bwd_Segment_Size_Avg', 'Packet_Length_Std', 'Packet_Length_Max', 'Packet_Length_Min', 'Bwd_Packet_Length_Max', 'Fwd_Header_Length', 'Packet_Length_Variance', 'Average_Packet_Size', 'Packet_Length_Mean', 'Fwd_Packet_Length_Min', 'Total_Length_of_Bwd_Packet', 'Fwd_IAT_Max', 'Fwd_Packet_Length_Max', 'Src_Port', 'FWD_Init_Win_Bytes', 'Fwd_IAT_Mean', 'Fwd_IAT_Total', 'Fwd_Packet_Length_Mean', 'Fwd_Segment_Size_Avg', 'Total_Length_of_Fwd_Packet', 'Subflow_Fwd_Bytes', 'Bwd_Init_Win_Bytes']

X = X[top_features]
X_normalized = sc.fit_transform(X)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

#Use cross validate to train the model and get the accuracy score, precision score, recall score, and f1 score
def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt ='g', xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Predicted Label", ylabel="True Label")

performance_metrics = ["test_f1_macro", "test_accuracy", "test_precision_macro", "test_recall_macro"]
foldCount = StratifiedKFold(10, shuffle=True, random_state=1)
cv_results_mlp = []
cv_results_xgb = []

#import XGBoost Classifier
from xgboost import XGBClassifier

#XGBoost Classifier
from sklearn.preprocessing import LabelEncoder
# create an instance of the LabelEncoder class
le = LabelEncoder()
# fit and transform the target variable
y = le.fit_transform(y)

xgb = XGBClassifier(verbosity=2, random_state=0, n_estimators=100, max_depth=10, learning_rate=0.3, gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor')

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cv_results = cross_validate(xgb, X_normalized, y, cv=foldCount, n_jobs=-1, verbose=1,
                                        scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'], return_estimator=True)


# Initialize an empty list to store the confusion matrices
confusion_matrices = []

# Loop through the fitted estimators
for estimator in cv_results['estimator']:
    # Make predictions using the current estimator
    y_pred = estimator.predict(X_normalized)

    # Calculate the confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Append the confusion matrix to the list
    confusion_matrices.append(cm)
    
#Combine the list of confusion_matrices into one matrix conf_mat
conf_mat = np.sum(confusion_matrices, axis=0)

cm = np.array(conf_mat)
# Classes
classes = ['Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'Streaming', 'VoIP']
title = "XGBoost Confusion Matrix (Model 2 - Darknet App Cat.)"
plot_matrix(cm, classes, title)
print(conf_mat)

#PRINT PERFORMANCE METRICS
cv_results_xgb.append(cv_results)
for i in performance_metrics:
    print(i)
    for j in cv_results_xgb[0][i]:
        print(j)
    
#XGB box plot
data1 = cv_results_xgb[0]['test_f1_macro']
data2 = cv_results_xgb[0]['test_accuracy']
data3 = cv_results_xgb[0]['test_precision_macro']
data4 = cv_results_xgb[0]['test_recall_macro']
allData = [data1,data2,data3,data4]

################### LEAVE THE BELOW UNCOMMENTED. IT PRODUCES THE BOXPLOT.
################### ALL THAT NEEDS TO BE SPECIFIED IS THE PLT.TITLE

### PLOT GENERATOR 
sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(8,6))
allData = [data1,data2,data3,data4]
g = sns.boxplot(data=allData, width=0.7)

###TITLE SETTING TEMPLATES (ONLY ONE SHOULD BE UNCOMMENTED AT A TIME.)
plt.title("XGBoost (Model 2 - Darknet App Cat.)", fontsize=16)

# X labels
xvalues = ["test_f1_macro", "test_accuracy", "test_precision_macro", "test_recall_macro"]

# x-labels
plt.xticks(np.arange(4), xvalues)

# setting y values
# plt.yticks(plt.yticks(np.arange(0,1,.1)))
plt.yticks(np.arange(0,1.1,.1))

### CHANGE ORDER #### ### CHANGE X coordinates ###, change median, change textstr

# Set colors of box plots 
palette= ['#B7C3D0','#B7C3D0','#B7C3D0','#B7C3D0','#FF6A6A']
color_dict = dict(zip(xvalues, palette))
for i in range(0,4):
    mybox = g.artists[i]
    mybox.set_facecolor(color_dict[xvalues[i]])
    
# F-Measure
median = round(data1.median(),1)
textstr = r"$\tilde {x}$" + f" = {median}"
g.text(-0.19, 1.1, textstr, fontsize=13,) #### delete bbox

# Accuracy
median = round(data2.median(),1)
textstr = r"$\tilde {x}$" + f" = {median}"
g.text(.81, 1.1, textstr, fontsize=13,)

# Precision
median = round(data3.median(),1)

textstr = r"$\tilde {x}$" + f" = {median}"
g.text(1.81, 1.1, textstr, fontsize=13,) 

# Recall
median = round(data4.median(),1)
textstr = r"$\tilde {x}$" + f" = {median}"
g.text(2.81, 1.1, textstr, fontsize=13)

#Plot
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt

#DIARY 
#There were no missing values in the dataset. 
#There were null values in the dataset: Index(['Flow_Bytes_per_second'], dtype='object')
#There were infinite values in the dataset: Index(['Flow_Bytes_per_second', 'Flow_Packets_per_second'], dtype='object')
#Removed 49 rows from the dataset because of null and infinite values
# Drop the categorical variables because they bias the dataset. 
# Normalized the inputs prior to feature ranking. 
# Collected top features (RFI, m = 10, n_est = 100)
# Trained model on features with importance greater than 0.01 

#Use the GPU if it's available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

df = pd.read_csv('C:/Users/ulixe/OneDrive/Desktop/Research/Darknet.csv')

#If the row contains an infinity value or a null value, remove it
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

#We are trying to detect whether it's a darknet connection or not.
df = df.drop(['Label_B'], axis=1)
X = df.drop(['Label_A'], axis=1)
y = df.Label_A

#Handle the categorical variables
categorical_variables = X.select_dtypes(include=['object']).columns
 
#drop the categorical variables 
X = X.drop(X[categorical_variables], axis=1)

# #Use label encoder to encode the target variable
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)
# # Normalize the features 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# X_normalized = sc.fit_transform(X)


# # Random forest importances
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_jobs=8, n_estimators=1000, max_depth=10, random_state=0)
# rf.fit(X_normalized, y)
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
# # top_features = []


# # Print the feature ranking with the name of feature. Only print those with importance greater than 0.01
top_features = ['Bwd_Packet_Length_Min', 'Flow_IAT_Min', 'Fwd_Seg_Size_Min', 'Fwd_Header_Length', 'Flow_Bytes_per_second',
                'Flow_IAT_Max', 'FWD_Init_Win_Bytes', 'Bwd_Packet_Length_Mean', 'Bwd_Packets_per_second', 'Flow_IAT_Mean',
                'Bwd_Segment_Size_Avg', 'Subflow_Bwd_Bytes', 'Flow_Packets_per_second', 'Fwd_Packets_per_second', 'Idle_Max',
                'Bwd_Init_Win_Bytes', 'Bwd_Packet_Length_Max', 'Average_Packet_Size', 'Packet_Length_Std',
                'Total_Length_of_Bwd_Packet', 'Fwd_Packet_Length_Min', 'Dst_Port', 'Packet_Length_Max',
                'Flow_Duration', 'Packet_Length_Variance', 'Packet_Length_Min', 'Packet_Length_Mean',
                'Idle_Mean', 'Total_Fwd_Packet', 'Fwd_IAT_Max', 'Fwd_IAT_Mean', 'Bwd_Header_Length']

# for f in range(X_normalized.shape[1]):
#     if importances[indices[f]] > 0.01:
#         print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], X.columns[indices[f]]))
#         top_features.append(X.columns[indices[f]])
# print(top_features)

#Only use top features
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

#import MLPClassifier from sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
#Create a MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(500, 250, 125, 60, 30, 15, 7), max_iter=1000, verbose=True, random_state=0, solver="adam"
                    , learning_rate_init=0.001, activation="relu", learning_rate="adaptive", early_stopping=True)

# mlp.fit(X_normalized, y)
cv_results = cross_validate(mlp, X_normalized, y, cv=foldCount,
                                        scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'], 
                                        n_jobs=8,
                                        verbose=True)
cv_results_mlp.append(cv_results)


# Confusion Matrix
y_pred = cross_val_predict(mlp, X_normalized, y, cv=foldCount, n_jobs = 8, verbose = True)
conf_mat = confusion_matrix(y, y_pred)
cm = np.array(conf_mat)
# Classes
classes = ['Benign', 'Darknet']
title = "Multi Layer Perceptron (Model 1 - DETECTION) Confusion Matrix"
plot_matrix(cm, classes, title)
print(conf_mat)

#PRINT PERFORMANCE METRICS
for i in performance_metrics:
    print(i)
    for j in cv_results_mlp[0][i]:
        print(j)
    
#MLP BoxPlot
data1 = cv_results_mlp[0]['test_f1_macro']
data2 = cv_results_mlp[0]['test_accuracy']
data3 = cv_results_mlp[0]['test_precision_macro']
data4 = cv_results_mlp[0]['test_recall_macro']
allData = [data1,data2,data3,data4]

################### LEAVE THE BELOW UNCOMMENTED. IT PRODUCES THE BOXPLOT.
################### ALL THAT NEEDS TO BE SPECIFIED IS THE PLT.TITLE

### PLOT GENERATOR 
sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(8,6))
allData = [data1,data2,data3,data4]
g = sns.boxplot(data=allData, width=0.7)

###TITLE SETTING TEMPLATES (ONLY ONE SHOULD BE UNCOMMENTED AT A TIME.)
plt.title("Multi Layer Perceptron (Model 1 - DETECTION)", fontsize=16)

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
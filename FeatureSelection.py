# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 17:05:51 2022

@author: ulixe
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

#from sklearn.gaussian_process.kernels import ConstantKernel, RBF    # for Gaussian process
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy as cp
#import graphviz
import pickle
from typing import Tuple
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import seaborn as sns; sns.set_theme()
sns.set(font_scale=1)

def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt ='g', xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Predicted Label", ylabel="True Label")

performance_metrics = ["test_f1_macro", "test_accuracy", "test_precision_macro", "test_recall_macro"]
foldCount = StratifiedKFold(10, shuffle=True, random_state=1)


cv_results_dt = []
cv_results_rf = []
cv_results_gnb = []
cv_results_svc = []
cv_results_mlp = []
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# data = pd.read_csv(r"C:\Users\ulixe\OneDrive\Desktop\Research\Darknet - DARKNET.csv")
data = pd.read_csv(r"C:\Users\ulixe\OneDrive\Desktop\Research\Darknet - BENIGN.csv")
# data = pd.read_csv(r"C:\Users\ulixe\OneDrive\Desktop\Research\Darknet - 2Labels.csv")
print(data.head())

df = pd.DataFrame(data, columns= [
"Flow_Duration",
'Total_Fwd_Packet',
"Total_Bwd_packets",
"Total_Length_of_Fwd_Packet",
"Total_Length_of_Bwd_Packet",
"Fwd_Packet_Length_Max",
"Fwd_Packet_Length_Min",
"Fwd_Packet_Length_Mean",
"Fwd_Packet_Length_Std",
"Bwd_Packet_Length_Max",
"Bwd_Packet_Length_Min",
'Bwd_Packet_Length_Mean',
"Bwd_Packet_Length_Std",
"Flow_Packets_per_second",
"Flow_IAT_Mean",
"Flow_IAT_Std",
"Flow_IAT_Max",
'Flow_IAT_Min',
'Fwd_IAT_Total',
"Fwd_IAT_Mean",
"Fwd_IAT_Std",
'Fwd_IAT_Max',
"Fwd_IAT_Min",
"Bwd_IAT_Total",
"Bwd_IAT_Mean",
"Bwd_IAT_Std",
"Bwd_IAT_Max",
"Bwd_IAT_Min",
"Fwd_PSH_Flags",
"Bwd_PSH_Flags",
"Fwd_URG_Flags",
'Bwd_URG_Flags',
"Fwd_Header_Length",
"Bwd_Header_Length",
"Fwd_Packets_per_second",
"Bwd_Packets_per_second",
"Packet_Length_Min",
'Packet_Length_Max',
'Packet_Length_Mean',
"Packet_Length_Std",
"Packet_Length_Variance",
'FIN_Flag_Count',
"SYN_Flag_Count",
"RST_Flag_Count",
"PSH_Flag_Count",
"ACK_Flag_Count",
"URG_Flag_Count",
"CWE_Flag_Count",
"ECE_Flag_Count",
"Down_Up_Ratio",
"Average_Packet_Size",
'Fwd_Segment_Size_Avg',
"Bwd_Segment_Size_Avg",
"Fwd_Bytes_Bulk_Avg",
"Fwd_Packet_Bulk_Avg",
"Fwd_Bulk_Rate_Avg",
"Bwd_Bytes_Bulk_Avg",
'Bwd_Packet_Bulk_Avg',
'Bwd_Bulk_Rate_Avg',
"Subflow_Fwd_Packets",
"Subflow_Fwd_Bytes",
'Subflow_Bwd_Packets',
"Subflow_Bwd_Bytes",
"FWD_Init_Win_Bytes",
"Bwd_Init_Win_Bytes",
"Fwd_Act_Data_Pkts",
"Fwd_Seg_Size_Min",
"Active_Mean",
"Active_Std",
"Active_Max",
"Active_Min",
'Idle_Mean',
"Idle_Std",
"Idle_Max",
"Idle_Min",
'Label_B'
# 'Label_A'

])
print(df)
inputs = df.drop('Label_B',axis='columns')
target = df['Label_B']
X = df.drop('Label_B',axis='columns')
Y = df['Label_B']

# inputs = df.drop('Label_A',axis='columns')
# target = df['Label_A']
# X = df.drop('Label_A',axis='columns')
# Y = df['Label_A']


# Principal Component Analysis
# Step1: Standardize the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(inputs)
input_scaled = scaler.transform(inputs)

# Step 2: Calculate the covariance matrix

cov_matrix = np.cov(input_scaled.T)

# Step 3: Calculate eigenvalues and eigenvectors

eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Step 4: Sort eigenvalues in descending order

idx = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:, idx]

# Step 5: Select the top 20 features

n_eig_vecs = eig_vecs[:, :20]

# Step 6: Transform the data

transformed_data = np.dot(input_scaled, n_eig_vecs)

# Using the top 20 features to train an MLP

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(transformed_data, target, test_size=0.5, random_state=0)

# Train the MLP
mlp = MLPClassifier(solver='adam', alpha=.05, activation='relu', max_iter=2000, hidden_layer_sizes=(100, 100),
                    random_state=1)
mlp.fit(X_train, y_train)

# Evaluate the MLP
cv_results = cross_validate(mlp, X_test, y_test, cv=foldCount,
                            scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])
cv_results_mlp.append(cv_results)

# Confusion Matrix
y_pred = cross_val_predict(mlp, X_test, y_test, cv=foldCount)
conf_mat = confusion_matrix(y_test, y_pred)
cm = np.array(conf_mat)

classes = ['Audio Streaming', 'Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'VoIP', 'Video Streaming']
title = "Multi Layer Perceptron (TOP 20) Confusion Matrix (DARKNET TRAFFIC - PCA)"
plot_matrix(cm, classes, title)
print(conf_mat)

# Print performance metrics
for i in performance_metrics:
    print(i)
    for j in cv_results_mlp[0][i]:
        print(j)

# MLP BoxPlot
data1 = cv_results_mlp[0]['test_f1_macro']
data2 = cv_results_mlp[0]['test_accuracy']
data3 = cv_results_mlp[0]['test_precision_macro']
data4 = cv_results_mlp[0]['test_recall_macro']
allData = [data1, data2, data3, data4]

### PLOT GENERATOR
sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))
allData = [data1, data2, data3, data4]
g = sns.boxplot(data=allData, width=0.7)

###TITLE SETTING TEMPLATES (ONLY ONE SHOULD BE UNCOMMENTED AT A TIME.)
# plt.title("Decision Tree (TOP 20) (Darknet Traffic)", fontsize=16)
# plt.title("Random Forest (TOP 20) (Darknet Traffic)", fontsize=16)
# plt.title("Gaussian Naive Bayes (TOP 20) (Darknet Traffic)", fontsize=16)
# plt.title("LinearSVC (TOP 20) (Darknet Traffic)", fontsize=16)
plt.title("Multi Layer Perceptron (TOP 20) (Darknet Traffic - PCA)", fontsize=16)

# X labels
xvalues = ["test_f1_macro", "test_accuracy", "test_precision_macro", "test_recall_macro"]

# x-labels
plt.xticks(np.arange(4), xvalues)

# setting y values
# plt.yticks(plt.yticks(np.arange(0,1,.1)))
plt.yticks(np.arange(0, 1.1, .1))

### CHANGE ORDER #### ### CHANGE X coordinates ###, change median, change textstr

# Set colors of box plots
palette = ['#B7C3D0', '#B7C3D0', '#B7C3D0', '#B7C3D0', '#FF6A6A']
color_dict = dict(zip(xvalues, palette))
for i in range(0, 4):
    mybox = g.artists[i]
    mybox.set_facecolor(color_dict[xvalues[i]])

# F-Measure
median = round(data1.median(), 1)
textstr = r"$\tilde {x}$" + f" = {median}"
g.text(-0.19, 1.1, textstr, fontsize=13, )  #### delete bbox

# Accuracy
median = round(data2.median(), 1)
textstr = r"$\tilde {x}$" + f" = {median}"
g.text(.81, 1.1, textstr, fontsize=13, )

# Precision
median = round(data3.median(), 1)

textstr = r"$\tilde {x}$" + f" = {median}"
g.text(1.81, 1.1, textstr, fontsize=13, )

# Recall
median = round(data4.median(), 1)
textstr = r"$\tilde {x}$" + f" = {median}"
g.text(2.81, 1.1, textstr, fontsize=13)

# Plot
plt.tight_layout()
plt.show()

#RFE w/ Random Forest
# # Create an instance of a Random Forest classifier
# estimator = RandomForestClassifier(n_estimators=75)
# # Create an RFE selector
# selector = RFE(estimator, n_features_to_select=20)
# # Fit the selector to the data
# selector = selector.fit(inputs, target)
# # Get the indices of the selected features
# selected_features = selector.get_support(indices=True)
# # Print the selected features
# print(inputs.columns[selected_features])

# #Random Forest
# model = RandomForestClassifier(n_estimators=75)
# model.fit(inputs, target)
# importances = model.feature_importances_
# final_df = pd.DataFrame({"Features": pd.DataFrame(inputs).columns, "Importances":importances})
# final_df.set_index("Importances")
# # final_df = final_df.sort_values("Importances")
# final_df = final_df.sort_values("Importances", ascending=False)
# final_df.plot.bar(color = 'teal')
# print(final_df.to_string())

# #Extreme Tree Classifier
# model = ExtraTreesClassifier()
# model.fit(X, Y)
# print(model.feature_importances_)
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(20).plot.bar()
# plt.show()
# list1=feat_importances.keys().to_list()
# print(feat_importances.nlargest(20))
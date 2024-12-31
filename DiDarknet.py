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

# #################### ONLY ONE DATA SET CAN BE UNCOMMENTED AT A TIME 

##2Label (DB) Data Set
data = pd.read_csv(r"C:\Users\ulixe\OneDrive\Desktop\Research\Darknet - 2Labels.csv")

# ##RANKED TOP 15 FEATURES (FROM BEST TO LEAST BEST). 
# ##COMMENT OUT BOTTOM 5-10 TO GET CLASSIFICATION RESULTS FOR EACH TOP 10/TOP 5 
df = pd.DataFrame(data, columns= ['Packet_Length_Max',
'Bwd_Packet_Length_Max',
"Packet_Length_Mean",
"Flow_IAT_Max",
'Average_Packet_Size',
"Flow_Duration",
"Flow_IAT_Min",
"Packet_Length_Std",
"Packet_Length_Variance",
"Bwd_Segment_Size_Avg",
"Bwd_Packet_Length_Mean",
"Fwd_Header_Length",
"Total_Length_of_Bwd_Packet",
"Bwd_Packet_Length_Min",
'Bwd_Packets_per_second', 
# "Flow_Packets_per_second",
# "Bwd_Header_Length",
# "Fwd_Packets_per_second",
# "Total_Length_of_Fwd_Packet",
# "Flow_IAT_Mean",
'Label_A'])
print(df)
inputs = df.drop('Label_A',axis='columns')
target = df['Label_A']



# ##3Label (VTB) Data Set
# data = pd.read_csv(r"C:\Users\ulixe\OneDrive\Desktop\Research\Darknet - 3Labels.csv")

# ##RANKED TOP 15 FEATURES (FROM BEST TO LEAST BEST). 
# ##COMMENT OUT BOTTOM 5-10 TO GET CLASSIFICATION RESULTS FOR EACH TOP 10/TOP 5 
# df = pd.DataFrame(data, columns= ['Packet_Length_Max',
# "Bwd_Packet_Length_Max",
# "Packet_Length_Mean",
# "Average_Packet_Size",
# "Flow_IAT_Min",
# "Packet_Length_Variance",
# "Packet_Length_Std",
# "Flow_IAT_Max",
# "Flow_Duration",
# "Bwd_Segment_Size_Avg",
# # "Bwd_Packet_Length_Mean",
# # "Total_Length_of_Bwd_Packet",
# # "Fwd_Header_Length",
# # "Total_Length_of_Fwd_Packet",
# # 'Bwd_Header_Length', 
# # "Bwd_Packets_per_second",
# # "Bwd_Packet_Length_Min",
# # "Flow_Packets_per_second",
# # "Fwd_Packet_Length_Max",
# # "Flow_IAT_Mean",
# 'Label_A'])
# print(df)

# inputs = df.drop('Label_A',axis='columns')
# target = df['Label_A']



################### ONLY ONE MODEL CAN BE UNCOMMENTED AT A TIME 


# ## Decision Tree
# dt = tree.DecisionTreeClassifier(random_state=0, max_depth=20)
# cv_results = cross_validate(dt, inputs, target, cv=foldCount,
#                                          scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])

# cv_results_dt.append(cv_results)


# # Confusion Matrix
# y_pred = cross_val_predict(dt, inputs, target, cv=foldCount)
# conf_mat = confusion_matrix(target, y_pred)
# cm = np.array(conf_mat)
# # Classes
# # classes = ['Benign', 'Tor', 'VPN'] #VTB DATABASE
# classes = ['Benign', 'Darknet'] #DB DATABASE
# title = "Decision Tree (TOP 5) Confusion Matrix (DB)"
# plot_matrix(cm, classes, title)
# print(conf_mat)


# # #Alternative To Heat Map (IF NEEDED)
# # count = 0
# # for i in conf_mat:
# #     print("")
# #     if (count <= len(classes)):
# #         print(classes[count])
# #         count += 1
# #     for j in i:
# #         print(j)

# #Check if Confusion Matrix Matches Results
# print("")
# print("THE MEAN IS:")
# print("")
# if len(conf_mat) == 3:
#     mean = (conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]) / 141530 
#     print(mean)
# else:
#     mean = (conf_mat[0][0] + conf_mat[1][1]) / 141530
#     print(mean)

# #PRINT PERFORMANCE METRICS
# for i in performance_metrics:
#     print(i)
#     for j in cv_results_dt[0][i]:
#         print(j)


# #DT box plot
# data1 = cv_results_dt[0]['test_f1_macro']
# data2 = cv_results_dt[0]['test_accuracy']
# data3 = cv_results_dt[0]['test_precision_macro']
# data4 = cv_results_dt[0]['test_recall_macro']
# allData = [data1,data2,data3,data4]




##Random Forest

rf = RandomForestClassifier()
# rf = RandomForestClassifier(random_state=0, max_depth=8)
cv_results = cross_validate(rf, inputs, target, cv=foldCount,
                                        scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])

# Confusion Matrix
y_pred = cross_val_predict(rf, inputs, target, cv=foldCount)
conf_mat = confusion_matrix(target, y_pred)
cm = np.array(conf_mat)
# Classes
# classes = ['Benign', 'Tor', 'VPN'] #VTB DATABASE
classes = ['Benign', 'Darknet'] #DB DATABASE
title = "Random Forest (TOP 5) Confusion Matrix (DB)"
plot_matrix(cm, classes, title)
print(conf_mat)

# #Alternative To Heat Map (IF NEEDED)
# count = 0
# for i in conf_mat:
#     print("")
#     if (count <= len(classes)):
#         print(classes[count])
#         count += 1
#     for j in i:
#         print(j)

#Check if Confusion Matrix Matches Results
print("")
print("THE MEAN IS:")
print("")
if len(conf_mat) == 3:
    mean = (conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]) / 141530 
    print(mean)
else:
    mean = (conf_mat[0][0] + conf_mat[1][1]) / 141530
    print(mean)

#PRINT PERFORMANCE METRICS
cv_results_rf.append(cv_results)
for i in performance_metrics:
    print(i)
    for j in cv_results_rf[0][i]:
        print(j)
    
#RF box plot
data1 = cv_results_rf[0]['test_f1_macro']
data2 = cv_results_rf[0]['test_accuracy']
data3 = cv_results_rf[0]['test_precision_macro']
data4 = cv_results_rf[0]['test_recall_macro']
allData = [data1,data2,data3,data4]





# ## Gaussian Naive Bayes
# X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=.5, random_state=0)
# gnb = GaussianNB()
# cv_results = cross_validate(gnb, inputs, target, cv=foldCount,
#                                         scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])
# cv_results_gnb.append(cv_results)      


# # Confusion Matrix
# y_pred = cross_val_predict(gnb, inputs, target, cv=foldCount)
# conf_mat = confusion_matrix(target, y_pred)
# cm = np.array(conf_mat)
# # Classes
# # classes = ['Benign', 'Tor', 'VPN'] #VTB DATABASE
# classes = ['Benign', 'Darknet'] #DB DATABASE
# title = "Gaussian Naive Bayes (TOP 5) Confusion Matrix (DB)"
# plot_matrix(cm, classes, title)
# print(conf_mat)

# # #Alternative To Heat Map (IF NEEDED)
# # count = 0
# # for i in conf_mat:
# #     print("")
# #     if (count <= len(classes)):
# #         print(classes[count])
# #         count += 1
# #     for j in i:
# #         print(j)
        
# #Check if Confusion Matrix Matches Results
# print("")
# print("THE MEAN IS:")
# print("")
# if len(conf_mat) == 3:
#     mean = (conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]) / 141530 
#     print(mean)
# else: 
#     mean = (conf_mat[0][0] + conf_mat[1][1]) / 141530
#     print(mean)

# #PRINT PERFORMANCE METRICS
# for i in performance_metrics:
#     print(i)
#     for j in cv_results_gnb[0][i]:
#         print(j)

# #gnb box plot
# data1 = cv_results_gnb[0]['test_f1_macro']
# data2 = cv_results_gnb[0]['test_accuracy']
# data3 = cv_results_gnb[0]['test_precision_macro']
# data4 = cv_results_gnb[0]['test_recall_macro']
# allData = [data1,data2,data3,data4]





# ###LinearSVC
# ###Train and score clf model with 5-fold cross validation
# clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, dual=False, tol=1e-3, verbose=1, max_iter=5000))
# cv_results = cross_validate(clf, inputs, target, cv=foldCount,
#                                         scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])
# cv_results_svc.append(cv_results)   
 

# # Confusion Matrix
# y_pred = cross_val_predict(clf, inputs, target, cv=foldCount)
# conf_mat = confusion_matrix(target, y_pred)
# cm = np.array(conf_mat)
# # Classes
# # classes = ['Benign', 'Tor', 'VPN'] #VTB DATABASE
# classes = ['Benign', 'Darknet'] #DB DATABASE
# title = "LinearSVC (TOP 5) Confusion Matrix (DB)"
# plot_matrix(cm, classes, title)
# print(conf_mat)

# # #Alternative To Heat Map (IF NEEDED)
# # count = 0
# # for i in conf_mat:
# #     print("")
# #     if (count <= len(classes)):
# #         print(classes[count])
# #         count += 1
# #     for j in i:
# #         print(j)
# #Check if Confusion Matrix Matches Results
# print("")
# print("THE MEAN IS:")
# print("")
# if len(conf_mat) == 3:
#     mean = (conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]) / 141530 
#     print(mean)
# else:
#     mean = (conf_mat[0][0] + conf_mat[1][1]) / 141530
#     print(mean)

# ##PRINT PERFORMANCE METRICS
# for i in performance_metrics:
#     print(i)
#     for j in cv_results_svc[0][i]:
#         print(j)
    
# ##SVC box plot
# data1 = cv_results_svc[0]['test_f1_macro']
# data2 = cv_results_svc[0]['test_accuracy']
# data3 = cv_results_svc[0]['test_precision_macro']
# data4 = cv_results_svc[0]['test_recall_macro']
# allData = [data1,data2,data3,data4]





# ### Multi Layer Perceptron (SKlearn)
            
# scaler = StandardScaler().fit(inputs)
# input_scaled = scaler.transform(inputs)
# X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.5, random_state=0)
# mlp = MLPClassifier(solver='adam', alpha=.05, activation='relu', max_iter=2000,
#                                 hidden_layer_sizes=(400, 400, 400, 300, 200, 100, 20), random_state=1)
# mlp.fit(inputs, target)
# cv_results = cross_validate(mlp, input_scaled, target, cv=foldCount,
#                                         scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])
# cv_results_mlp.append(cv_results)


# # Confusion Matrix
# y_pred = cross_val_predict(mlp, input_scaled, target, cv=foldCount)
# conf_mat = confusion_matrix(target, y_pred)
# cm = np.array(conf_mat)
# # Classes
# # classes = ['Benign', 'Tor', 'VPN'] #VTB DATABASE
# classes = ['Benign', 'Darknet'] #DB DATABASE
# title = "Multi Layer Perceptron (TOP 15) Confusion Matrix (DB)"
# plot_matrix(cm, classes, title)
# print(conf_mat)

# # #Alternative To Heat Map (IF NEEDED)
# # count = 0
# # for i in conf_mat:
# #     print("")
# #     if (count <= len(classes)):
# #         print(classes[count])
# #         count += 1
# #     for j in i:
# #         print(j)

# #Check if Confusion Matrix Matches Results
# print("")
# print("THE MEAN IS:")
# print("")
# if len(conf_mat) == 3:
#     mean = (conf_mat[0][0] + conf_mat[1][1] + conf_mat[2][2]) / 141530 
#     print(mean)
# else:
#     mean = (conf_mat[0][0] + conf_mat[1][1]) / 141530
#     print(mean)

# #PRINT PERFORMANCE METRICS
# for i in performance_metrics:
#     print(i)
#     for j in cv_results_mlp[0][i]:
#         print(j)
    
# #MLP BoxPlot
# data1 = cv_results_mlp[0]['test_f1_macro']
# data2 = cv_results_mlp[0]['test_accuracy']
# data3 = cv_results_mlp[0]['test_precision_macro']
# data4 = cv_results_mlp[0]['test_recall_macro']
# allData = [data1,data2,data3,data4]




#################### LEAVE THE BELOW UNCOMMENTED. IT PRODUCES THE BOXPLOT.
#################### ALL THAT NEEDS TO BE SPECIFIED IS THE PLT.TITLE



### PLOT GENERATOR 
sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(8,6))
allData = [data1,data2,data3,data4]
g = sns.boxplot(data=allData, width=0.7)

###TITLE SETTING TEMPLATES (ONLY ONE SHOULD BE UNCOMMENTED AT A TIME.)
# plt.title("Decision Tree (TOP 5) (DB)", fontsize=16)
# plt.title("Random Forest (TOP 5) (DB)", fontsize=16)
# plt.title("Gaussian Naive Bayes (TOP 5) (DB)", fontsize=16)
# plt.title("LinearSVC (TOP 5) (DB)", fontsize=16)
plt.title("Multi Layer Perceptron (TOP 15) (DB)", fontsize=16)

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
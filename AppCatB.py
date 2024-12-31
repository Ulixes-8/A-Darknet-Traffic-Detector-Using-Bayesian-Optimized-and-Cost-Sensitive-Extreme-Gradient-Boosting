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

#Performance Metrics
performance_metrics = ["test_f1_macro", "test_accuracy", "test_precision_macro", "test_recall_macro"]
cv_results_mlp = []
cv_results_xgb = []

#Stratified K-Fold
foldCount = StratifiedKFold(10, shuffle=True, random_state=1)



### DATA PREPROCESSING ###
#Import the dataset
df = pd.read_csv('C:/Users/ulixe/OneDrive/Desktop/Research/Darknet.csv')

#Remove null and infinite values from the dataset
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

#We are trying to determine the application given the benign traffic
df = df[df.Label_A != 'Darknet']
df = df.drop(['Label_A'], axis=1)
X = df.drop(['Label_B'], axis=1)
y = df.Label_B

#Handle the categorical variables
categorical_variables = X.select_dtypes(include=['object']).columns
#drop the categorical variables 
X = X.drop(X[categorical_variables], axis=1)

#Normalize the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#Only use top features
# top_features = ['Dst_Port', 'Idle_Max', 'Src_Port', 'Fwd_IAT_Max', 'Fwd_IAT_Total', 'Idle_Mean', 'Flow_IAT_Max', 'Idle_Min', 'Fwd_IAT_Mean', 'Fwd_Header_Length', 'FWD_Init_Win_Bytes', 'Packet_Length_Min', 'Fwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean', 'Subflow_Bwd_Bytes', 'Flow_IAT_Mean', 'Bwd_Init_Win_Bytes', 'Fwd_Packets_per_second', 'Bwd_Segment_Size_Avg', 'Flow_Duration', 'Fwd_Seg_Size_Min', 'Flow_Packets_per_second', 'Flow_IAT_Min', 'Bwd_Packets_per_second', 'Average_Packet_Size', 'Flow_Bytes_per_second', 'Packet_Length_Mean', 'Packet_Length_Max', 'Bwd_Packet_Length_Max', 'Fwd_Packet_Length_Max', 'Total_Length_of_Fwd_Packet', 'Fwd_Segment_Size_Avg', 'Bwd_Packet_Length_Min', 'Packet_Length_Std', 'Fwd_IAT_Min', 'Total_Length_of_Bwd_Packet', 'Packet_Length_Variance', 'Fwd_Packet_Length_Mean', 'Bwd_Header_Length']
# X = X[top_features]
X_normalized = sc.fit_transform(X)



###DATA BALANCING###

#Before balancing the dataset
classes = ['Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'Streaming', 'VoIP']
count = y.value_counts()
class_counts = pd.DataFrame({'Class': classes, 'Count': count}).sort_values('Class')
plt.bar(class_counts['Class'], class_counts['Count'])
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Counts of Each Class Before Balancing')
plt.show()

#Balance the dataset 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# #UNDERS SAMPLING
# undersampler = RandomUnderSampler(random_state = 42)
# X_normalized, y = undersampler.fit_resample(X, y)

# ##OVER SAMPLING
# oversampler = RandomOverSampler(random_state = 42, sampling_strategy = 'not majority')
# X_normalized, y = oversampler.fit_resample(X_normalized, y)

# # #SMOTE
# from imblearn.over_sampling import SMOTE
# smote = SMOTE(random_state = 42, sampling_strategy = 'not majority')
# X_normalized, y = smote.fit_resample(X_normalized, y)

# # #ADASYN
# from imblearn.over_sampling import ADASYN
# adasyn = ADASYN(random_state = 42, sampling_strategy = 'not majority')
# X_normalized, y = adasyn.fit_resample(X_normalized, y)
        

#SMOTEENN
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=42, sampling_strategy = 'not majority')
X_normalized, y = smote_enn.fit_resample(X_normalized, y)

# # #SMOTETomek
# from imblearn.combine import SMOTETomek
# smote_tomek = SMOTETomek(random_state=42, sampling_strategy = 'not majority')
# X_normalized, y = smote_tomek.fit_resample(X_normalized, y)


#After balancing the dataset
count = y.value_counts()
print(count)
class_counts = pd.DataFrame({'Class': classes, 'Count': count}).sort_values('Class')
plt.bar(class_counts['Class'], class_counts['Count'])
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Counts of Each Class After Balancing')
plt.show()


###MODEL TRAINING###

#import XGBoost Classifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

xgb = XGBClassifier(verbosity=2, random_state=0, n_estimators=100, max_depth=10, learning_rate=0.35, gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor')

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
title = "XGBoost Confusion Matrix (Model 3 - Benign App Cat.)"
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



###BOX PLOT GENERATOR###
sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(8,6))
allData = [data1,data2,data3,data4]
g = sns.boxplot(data=allData, width=0.7)

###TITLE SETTING TEMPLATES (ONLY ONE SHOULD BE UNCOMMENTED AT A TIME.)
# plt.title("Multi Layer Perceptron (Model 3 - Benign App Cat.)", fontsize=16)
plt.title("XGBoost (Model 3 - Benign App Cat.)", fontsize=16)

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
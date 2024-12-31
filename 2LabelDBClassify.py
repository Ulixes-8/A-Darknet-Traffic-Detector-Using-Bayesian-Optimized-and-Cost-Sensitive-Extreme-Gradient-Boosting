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

import numpy as np
import pandas as pd
#import graphviz
import pickle

foldCount = 10


cv_results_dt = []
cv_results_rf = []
cv_results_gnb = []
cv_results_svc = []
cv_results_mlp = []




##2Label (DB) Data Set
data = pd.read_csv(r"C:\Users\ulixe\OneDrive\Desktop\Research\Darknet - 2Labels.csv")

##RANKED TOP 15 FEATURES (FROM BEST TO LEAST BEST). 
##COMMENT OUT BOTTOM 5-10 TO GET CLASSIFICATION RESULTS FOR EACH TOP 10/TOP 5 
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
'Label_A'])
#print(df)
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
# "Bwd_Packet_Length_Mean",
# "Total_Length_of_Bwd_Packet",
# "Fwd_Header_Length",
# "Total_Length_of_Fwd_Packet",
# 'Bwd_Header_Length', 
# 'Label_A'])
# print(df)

# inputs = df.drop('Label_A',axis='columns')
# target = df['Label_A']









### Decision Tree
dt = tree.DecisionTreeClassifier(random_state=0, max_depth=5)
cv_results = cross_validate(dt, inputs, target, cv=foldCount,
                                        scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])
print(cv_results)
cv_results_dt.append(cv_results)

# Plot tree
dt = dt.fit(inputs, target)
# tree.plot_tree(dt)
print(dt.score(inputs, target))
target




###Random Forest
# rf = RandomForestClassifier(random_state=0, max_depth=5)
# cv_results = cross_validate(rf, inputs, target, cv=foldCount,
#                                         scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])
# print(cv_results)
# cv_results_rf.append(cv_results)






#             ## Gaussian Naive Bayes
# X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.5, random_state=0)
# gnb = GaussianNB()
# cv_results = cross_validate(gnb, inputs, target, cv=foldCount,
#                                         scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])
# print(cv_results)
# cv_results_gnb.append(cv_results)
            







            ## LinearSVC
#             # Train and score clf model with 5-fold cross validation
# clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, verbose=1, max_iter=250000))
# cv_results = cross_validate(clf, inputs, target, cv=foldCount,
#                                         scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])
# sorted(cv_results.keys())
# print(cv_results.keys())
# cv_results_svc.append(cv_results)   









#             #             ### Multi Layer Perceptron (SKlearn)
            
# scaler = StandardScaler().fit(inputs)
# input_scaled = scaler.transform(inputs)
# #print(ids2018_csv_file[i])


# X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.5, random_state=0)
# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='relu', max_iter=20000,
#                                 hidden_layer_sizes=(15, 5, 2), random_state=1)
#             #mlp.fit(inputs, target)

# cv_results = cross_validate(mlp, input_scaled, target, cv=foldCount,
#                                         scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])
# print(cv_results)
# cv_results_mlp.append(cv_results)
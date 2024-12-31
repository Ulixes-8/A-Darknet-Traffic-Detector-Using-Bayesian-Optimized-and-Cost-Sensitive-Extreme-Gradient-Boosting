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

classes = ['Benign', 'Darknet']

### DATA PREPROCESSING ###
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

print(X.shape)

#drop the columns with only 1 unique value
X = X.loc[:, X.nunique() != 1]
print(X.shape)

print("Number of negative class: ", len(y[y==0]))
print("Number of positive class: ", len(y[y==1]))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Train-Test Split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#Standard scaler
# scaler = StandardScaler()

#min-max scaler
scaler = MinMaxScaler()

# Fit the scaler using the training data
scaler.fit(X_train) #Fit the scaler only on the training data

# Transform the training, validation, and test sets using the same scaler
#This is to prevent data leakage. Information from the test set should not be used to transform the training set
#Data leakage is when information from outside the test and validation set is used to create the model  
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#print the min, mean, and max of the training set
print(f"Min: {X_train.min():.4f}, Mean: {X_train.mean():.4f}, Max: {X_train.max():.4f}")

#plot distribution of the training set
sns.displot(X_train[:,0], kde=False, rug=False, bins=20)
plt.show()


top_features = ['Bwd_Packet_Length_Min', 'Flow_IAT_Min', 'Fwd_Seg_Size_Min', 'Fwd_Header_Length', 'Flow_Bytes_per_second',
                'Flow_IAT_Max', 'FWD_Init_Win_Bytes', 'Bwd_Packet_Length_Mean', 'Bwd_Packets_per_second', 'Flow_IAT_Mean',
                'Bwd_Segment_Size_Avg', 'Subflow_Bwd_Bytes', 'Flow_Packets_per_second', 'Fwd_Packets_per_second', 'Idle_Max',
                'Bwd_Init_Win_Bytes', 'Bwd_Packet_Length_Max', 'Average_Packet_Size', 'Packet_Length_Std',
                'Total_Length_of_Bwd_Packet', 'Fwd_Packet_Length_Min', 'Dst_Port', 'Packet_Length_Max',
                'Flow_Duration', 'Packet_Length_Variance', 'Packet_Length_Min', 'Packet_Length_Mean',
                'Idle_Mean', 'Total_Fwd_Packet', 'Fwd_IAT_Max', 'Fwd_IAT_Mean', 'Bwd_Header_Length']

#import XGBoost Classifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
le.fit(y_train) #You must only fit the label encoder on the training data so the encodings are consistent across the training, validation, and test sets
y_train = le.transform(y_train)
y_test = le.transform(y_test)


scale_positive_weights = [96]

for i in range(len(scale_positive_weights)):
        
    xgb = XGBClassifier(verbosity=2, random_state=0, n_estimators=100, max_depth=10, learning_rate=0.35, gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor', scale_pos_weight = scale_positive_weights[i])
    xgb.fit(X_train,y_train)


    # #Visualize the training and validation loss over epochs
    # from yellowbrick.model_selection import LearningCurve
    # visualizer = LearningCurve(xgb, scoring='neg_log_loss', cv=5, n_jobs=-1, train_sizes=np.linspace(0.3, 1.0, 10))
    # visualizer.fit(X_train, y_train)
    # visualizer.show()

    # #Visualize the training and validation recall_macro over epochs
    # from yellowbrick.model_selection import ValidationCurve
    # visualizer = ValidationCurve(xgb, param_name="max_depth", param_range=np.arange(1, 11), cv=5, scoring="recall_macro", n_jobs=-1)
    # visualizer.fit(X_train, y_train)
    # visualizer.show()


    scoring_names = ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 
    'balanced_accuracy', 'brier_score_loss', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 
    'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 
    'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'log_loss', 'max_error', 'mutual_info_score', 'neg_brier_score', 
    'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 
    'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 
    'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 
    'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'v_measure_score']


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



    # Generate a confusion matrix
    conf_mat = confusion_matrix(y_test,y_pred)

    cm = np.array(conf_mat)
    # Classes
    title = "XGBoost Confusion Matrix (Model 1 - Darknet Detection)"

    plot_matrix(cm, classes, title)
    print(conf_mat)
    from sklearn.metrics import roc_auc_score, roc_curve

    y_pred_prob = xgb.predict_proba(X_test)[:, 1]  # The second column corresponds to the positive class (Darknet)
    auc_roc = roc_auc_score(y_test, y_pred_prob)

    print(f'AUC-ROC: {auc_roc:.4f}')

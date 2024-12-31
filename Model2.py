#Model 2: Darknet Application Categorization with F1 Bayesian Optimization 

# In[1] Data Preprocessing 

# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# define StratifiedKFold cross-validation object with 5 splits
foldCount = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# define a function to plot confusion matrices
def plot_matrix(cm, classes, title):
  ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt ='g', xticklabels=classes, yticklabels=classes, cbar=False)
  ax.set(title=title, xlabel="Predicted Label", ylabel="True Label")

# define list of class labels
classes = ['Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'Streaming', 'VoIP']

# read data from CSV file
df = pd.read_csv('C:/Users/ulixe/OneDrive/Desktop/Research/Darknet.csv')

# print the data shape before cleaning
print(f"data shape before cleaning: {df.shape}")

# replace infinite values with NaN and drop rows with NaN values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# print the data shape after cleaning
print(f"data shape after cleaning: {df.shape}")

#Remove all Benign rows from the dataset
df = df[df.Label_A != 'Benign']

#Remove the Label_A (D/B) column
df = df.drop(['Label_A'], axis=1)

# create X and y variables to store feature and target values, respectively
X = df.drop(['Label_B'], axis=1)
y = df.Label_B

# get categorical column names in X and drop them from X
categorical_variables = X.select_dtypes(include=['object']).columns
print(f"Columns to be dropped from X: {categorical_variables}")
X = X.drop(X[categorical_variables], axis=1)
print(f"X shape after dropping columns: {X.shape}")

# get column names with only one unique value and drop them from X
unique_columns = X.columns[X.nunique() == 1]
print(f"Columns to be dropped from X: {unique_columns}")
X = X.drop(unique_columns, axis=1)
print(f"X shape after dropping columns: {X.shape}")

# print the number of instances in each class
print(y.value_counts())

# split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Distribution of Classes
import matplotlib.pyplot as plt

# count the number of instances for each class
class_counts = {}
for item in y:
    if item in class_counts:
        class_counts[item] += 1
    else:
        class_counts[item] = 1

# sort the classes by count in descending order
sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

# create the bar chart
fig, ax = plt.subplots()
ax.bar(sorted_class_counts.keys(), sorted_class_counts.values())

# add the counts as text labels on the bars
for i, v in enumerate(sorted_class_counts.values()):
    ax.text(i, v + 0.5, str(v), ha='center')

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Classes')
plt.xticks(rotation=45)
plt.show()

#In[2] F1 Bayesian Optimization for the Random Forest Classifier/Feature Ranker

# import necessary libraries
from hyperopt import fmin, hp, tpe
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (f1_score, make_scorer,
                             precision_score, recall_score)

# define a parameter space for hyperparameter optimization
param_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 200, 1),
    'max_depth': hp.quniform('max_depth', 15, 20, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 5, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
}

# create an empty list to store search results
search_results = []

def objective1(params):
    # instantiate a random forest classifier with given hyperparameters
    rf = RandomForestClassifier(random_state=42, n_jobs=-1,
                        n_estimators=int(params['n_estimators']),
                        max_depth=int(params['max_depth']),
                        min_samples_split=int(params['min_samples_split']),
                        min_samples_leaf=int(params['min_samples_leaf']),
                    )
                    
    # create a custom scoring function using macro-averaged F1 score
    f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
    # perform cross-validation with the random forest classifier using the custom scorer
    scores = cross_val_score(rf, X_train, y_train, cv=foldCount, scoring=f1_scorer, n_jobs=-1)
    print(scores)
    # calculate the mean score across all folds
    f1 = scores.mean()
    print(f"Cross-validated F1 score: {f1}")
    # store the hyperparameters and corresponding score in search_results
    score = f1
    search_results.append((params, score))

    # return the negative score (to be minimized)
    return -score

# use Tree-structured Parzen Estimator (TPE) algorithm to minimize objective1 function
best = fmin(fn=objective1, space=param_space, algo=tpe.suggest, max_evals=10)

# print the best score found by the optimizer
print(f"Best score: {-objective1(best)}")

#In[3] Visualize the Hyperparameter Optimization Results using Pairplot
# import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# find the best score among all search results
best_score = max(search_results, key=lambda x: x[1])[1]
worst_score = min(search_results, key=lambda x: x[1])[1]

# print the best hyperparameters and corresponding score found by the optimizer
print(f"Best parameters: {best}")
print(f"Best score: {best_score}")
print(f"Worst score: {worst_score}")

# extract hyperparameters and corresponding scores from search_results
hyperparams = [result[0] for result in search_results]
scores = [result[1] for result in search_results]

# create a pandas dataframe with hyperparameters and scores
data = pd.DataFrame(hyperparams)
data['F1'] = scores

# create a pair plot to visualize the relationship between hyperparameters and scores
sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, aspect=1.2, height=3)

plt.show()

#In[4] Visualize the Hyperparameter Optimization Results using Parallel Coordinates Plot
 
import plotly.express as px

# Convert search_results to a pandas DataFrame
search_results_df = pd.DataFrame([result[0] for result in search_results])
search_results_df['f1'] = [result[1] for result in search_results]

# Create a parallel coordinates plot
fig = px.parallel_coordinates(
    search_results_df,
    dimensions=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'f1'],
    color='f1',
    labels={
        'n_estimators': 'Number of Estimators',
        'max_depth': 'Max Depth',
        'min_samples_split': 'Min Samples Split',
        'min_samples_leaf': 'Min Samples Leaf',
        'f1': 'F1 Score'
    },
    color_continuous_scale=px.colors.sequential.Turbo,
    range_color=[0.8948142008294436,0.9208574783316633],
)

fig.show()

#In[5] Build the Optimized Random Forest Classifier and Rank Features

# instantiate a random forest classifier with the best hyperparameters found by the optimizer
rf = RandomForestClassifier(random_state=42, n_jobs=-1,
                        n_estimators=int(best['n_estimators']),
                        max_depth=int(best['max_depth']),
                        min_samples_split=int(best['min_samples_split']),
                        min_samples_leaf=int(best['min_samples_leaf']),
                    )

# create an array to store feature importances
importances = np.zeros(X_train.shape[1])

# iterate through each fold in the cross-validation
for train_index, test_index in foldCount.split(X_train, y_train):
    # split data into training and testing sets
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    
    # train the random forest classifier on the training set
    rf.fit(X_train_fold, y_train_fold)
    
    # add feature importances to the array
    importances += rf.feature_importances_

# divide the importances by the number of folds to get average importances
importances /= foldCount.get_n_splits()

# get the feature names from X_train
feature_names = X_train.columns

# sort the feature importances in descending order
sorted_indices = np.argsort(importances)[::-1]

# get the top features based on their importances
top_features = [feature_names[i] for i in sorted_indices]

# print the feature names and their importances
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]}")
    
# the following lines of code are commented out. I used them to print the feature names and importances separately for easy pasting in excel. 
for i in sorted_indices:
    print(f"{feature_names[i]}")
    
for i in sorted_indices:
    print(f"{importances[i]}")
    
#In[6] Plot the Reverse Sorted Feature Importances using a Line Chart

# Get the feature importances and names
feature_importances = importances
feature_names = X_train.columns

# Sort the feature importances and names in descending order
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_importances = feature_importances[sorted_indices]
sorted_names = feature_names[sorted_indices]

# Plot the sorted feature importances
plt.figure(figsize=(12, 6))
plt.plot(sorted_names, sorted_importances, marker='o')
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.title("Reverse Sorted Line Chart of Feature Importances")
plt.show()

#In[7] F1 Bayesian Optimization for the final XGBoost Classifier 
#Number of features is a hyperparameter, so that feature ranking is important 
#We take the first N features 

# import necessary libraries
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# instantiate a label encoder to encode class labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# define a parameter space for hyperparameter optimization
param_space = {
    'n_estimators': hp.quniform('n_estimators', 100, 200, 1),
    'max_depth': hp.quniform('max_depth', 10, 20, 1),
    'learning_rate': hp.uniform('learning_rate', 0.1, 0.2),
    'max_bin' : hp.quniform('max_bin', 512, 1280, 1),
    'num_features': hp.quniform('num_features', 1, len(sorted_names), 1)
}

# create an empty list to store search results
search_results = []

# define an objective function to minimize
def objective(params):
    # extract the number of features to be used from hyperparameters
    num_features = int(params['num_features'])
    # get the names of the top features based on their importances
    selected_features = sorted_names[:num_features]
    # create a new training set with only the selected features
    X_train_selected = X_train[selected_features]
    # instantiate an XGBoost classifier with given hyperparameters
    clf = XGBClassifier(random_state=42, gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor', 
                        objective='multi:softmax', num_class=7,
                        eval_metric='mlogloss',
                        n_estimators=int(params['n_estimators']),
                        max_depth=int(params['max_depth']),
                        learning_rate=params['learning_rate'],
                        max_bin=int(params['max_bin']),
                    )
                    
    # create a custom scoring function using F1 score
    f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)
    
    # perform cross-validation with the random forest classifier using the custom scorer
    scores = cross_val_score(clf, X_train_selected, y_train, cv=foldCount, scoring=f1_scorer)
    print(scores)
    # calculate the mean score across all folds
    f1 = scores.mean()
    print(f"Cross-validated F1 score: {f1}")
    # store the hyperparameters and corresponding score in search_results
    score = f1
    search_results.append((params, score))

    # return the negative score (to be minimized)
    return -score


# use Tree-structured Parzen Estimator (TPE) algorithm to minimize objective function
best = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=10)

# print the best score found by the optimizer
print(f"Best score: {-objective(best)}")

#In[8] Visualization of Hyperparameter Search Results using a Pair Plot

# find the best score among all search results
best_score = max(search_results, key=lambda x: x[1])[1]
worst_score = min(search_results, key=lambda x: x[1])[1]
# print the best hyperparameters found by the optimizer
print(f"Best parameters: {best}")
print(f"Best score: {best_score}")
print(f"Worst score: {worst_score}")

# extract hyperparameters and corresponding scores from search_results
hyperparams = [result[0] for result in search_results]
scores = [result[1] for result in search_results]

# create a pandas dataframe with hyperparameters and scores
data = pd.DataFrame(hyperparams)
data['f1'] = scores

# create a pair plot to visualize the relationship between hyperparameters and scores
sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
plt.show()

#In[9] Visualize the Hyperparameter Optimization Results using Parallel Coordinates Plot
# Convert search_results to a pandas DataFrame
search_results_df = pd.DataFrame([result[0] for result in search_results])
search_results_df['f1'] = [result[1] for result in search_results]

# Create a parallel coordinates plot
fig = px.parallel_coordinates(
    search_results_df,
    dimensions=['n_estimators', 'max_depth', 'learning_rate', 'max_bin', 'num_features', 'f1'],
    color='f1',
    labels={
        'n_estimators': 'Number of Estimators',
        'max_depth': 'Max Depth',
        'learning_rate': 'Learning Rate',
        'max_bin': 'Max Bin',
        'num_features': 'Number of Features',
        'f1': 'F1 Score'
    },

    color_continuous_scale=px.colors.sequential.Turbo,
    range_color=[0.9230970477330917,0.942187700467699],
)

fig.show()

#In[10] Cross-Validated MACRO Scores on Training Data with Best Hyperparameters
#import accuracy score
from sklearn.metrics import accuracy_score

# Select the top num_features features for both X_train and X_test
print(f"Before: {X_train.shape}")
X_train = X_train[sorted_names[:int(best['num_features'])]]
print(f"After: {X_train.shape}")

print(f"Before: {X_test.shape}")
X_test = X_test[sorted_names[:int(best['num_features'])]]
print(f"After: {X_test.shape}")

# Convert X_train, X_test, and y_train to pandas dataframes
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)

# Create an XGBoost Classifier with the best hyperparameters
xgb = XGBClassifier(random_state=42, gpu_id=0, tree_method='gpu_hist', predictor='gpu_predictor', 
                        objective='multi:softmax', num_class=7,
                        eval_metric='mlogloss',
                        n_estimators=int(best['n_estimators']),
                        max_depth=int(best['max_depth']),
                        learning_rate=best['learning_rate'],
                        max_bin=int(best['max_bin']),
                    )

# Use the existing StratifiedKFold object for cross-validation
kf = foldCount

# Collect the macro scores for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for train_index, test_index in kf.split(X_train, y_train):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    xgb.fit(X_train_fold, y_train_fold)
    y_pred = xgb.predict(X_test_fold)
    accuracy_scores.append(accuracy_score(y_test_fold, y_pred))
    precision_scores.append(precision_score(y_test_fold, y_pred, average="macro"))
    recall_scores.append(recall_score(y_test_fold, y_pred, average="macro"))
    f1_scores.append(f1_score(y_test_fold, y_pred, average="macro"))

#In[11] Box Plot of Cross-Validated MACRO Scores on Training Data with Best Hyperparameters
# Create a box plot of the scores
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
fig.subplots_adjust(wspace=0.5)

score_names = ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1']
score_values = [accuracy_scores, precision_scores, recall_scores, f1_scores]

# Create a boxplot for each score
for i, (ax, score_name, score_value) in enumerate(zip(axs, score_names, score_values)):
    ax.boxplot(score_value)
    ax.set_title(score_name)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_ylim([0.8, 1])  # Set y-axis limits to [0, 1]
    ax.yaxis.grid(True)  # Add horizontal grid lines

# Show the boxplots
plt.show()

#Print the mean scores
print(f"Accuracy: {np.mean(accuracy_scores)}")
print(f"Macro Precision: {np.mean(precision_scores)}")
print(f"Macro Recall: {np.mean(recall_scores)}")
print(f"Macro F1: {np.mean(f1_scores)}")

#In[12] Weighted Average of the Cross-Validated Scores on Training Data with Best Hyperparameters

# Collect the scores for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for train_index, test_index in kf.split(X_train, y_train):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    xgb.fit(X_train_fold, y_train_fold)
    y_pred = xgb.predict(X_test_fold)
    accuracy_scores.append(accuracy_score(y_test_fold, y_pred))
    precision_scores.append(precision_score(y_test_fold, y_pred, average="weighted"))
    recall_scores.append(recall_score(y_test_fold, y_pred, average="weighted"))
    f1_scores.append(f1_score(y_test_fold, y_pred, average="weighted"))


#In[13] Create Box Plot of Cross-Validated WEIGHTED Scores on Training Data with Best Hyperparameters
# Create a box plot of the scores
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
fig.subplots_adjust(wspace=0.5)

score_names = ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1']
score_values = [accuracy_scores, precision_scores, recall_scores, f1_scores]

# Create a boxplot for each score
for i, (ax, score_name, score_value) in enumerate(zip(axs, score_names, score_values)):
    ax.boxplot(score_value)
    ax.set_title(score_name)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_ylim([0.8, 1])  # Set y-axis limits to [0, 1]
    ax.yaxis.grid(True)  # Add horizontal grid lines

# Show the boxplots
plt.show()

#Print the mean scores
print(f"Accuracy: {np.mean(accuracy_scores)}")
print(f"Weighted Precision: {np.mean(precision_scores)}")
print(f"Weighted Recall: {np.mean(recall_scores)}")
print(f"Weighted F1: {np.mean(f1_scores)}")

#In[14] Fit the XGBoost Classifier on the entire training set and predict on the test set

#Fit the model on the entire training set
xgb.fit(X_train, y_train)

#predict on test data
y_pred = xgb.predict(X_test)

#Get the performance metrics
accuracy = accuracy_score(y_test, y_pred)
macro_recall = recall_score(y_test, y_pred, average="macro")
macro_precision = precision_score(y_test, y_pred, average="macro")
macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"Accuracy (on test-set): {accuracy}\nMacro Recall (on test-set): {macro_recall}\nMacro Precision (on test-set): {macro_precision}\nMacro F1 (on test-set): {macro_f1}")

#In[15] Weigted Average of the Scores on the Test Set

#Get the performance metrics
weighted_recall = recall_score(y_test, y_pred, average="weighted")
weighted_precision = precision_score(y_test, y_pred, average="weighted")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Accuracy (on test-set): {accuracy}\nWeighted Recall (on test-set): {weighted_recall}\nWeighted Precision (on test-set): {weighted_precision}\nWeighted F1 (on test-set): {weighted_f1}")

#In[16] Plot the Confusion Matrix on the Test Set
from sklearn.metrics import confusion_matrix

#Create a confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
cm = np.array(conf_mat)
title = "XGBoost Confusion Matrix (Darknet Application Categorization)"

#Plot the confusion matrix
plot_matrix(cm, classes, title)
print(conf_mat)

#In[17] Produce a Classification Report on the Test Set
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport

#Generate a classification report using yellowbrick
visuzalizer = ClassificationReport(xgb, classes=classes, support=True, digits=5)
#Fit the training data to the visualizer
visuzalizer.fit(X_train, y_train)
#Evaluate the model on the test data
visuzalizer.score(X_test, y_test)
#Draw the visualization
visuzalizer.show()

#In[18] Produce a bar chart of the model performance on the test set (macro vs weighted)
# Define the labels and values to plot
labels = ['Accuracy', 'Weighted Precision', 'Macro Precision', 'Weighted Recall', 'Macro Recall', 'Weighted F1-score', 'Macro F1-score']
values = [round(accuracy, 4), round(weighted_precision, 4), round(macro_precision, 4), round(weighted_recall, 4), round(macro_recall, 4), round(weighted_f1, 4), round(macro_f1, 4)]

# Define the x-axis and width of the bars
x = np.arange(len(labels))
width = 0.35

# Create a new plot with the defined dimensions
fig, ax = plt.subplots(figsize=(10,6))

# Create a set of bars with the defined values
rects = ax.bar(x, values, width)

# Set the labels and title of the plot
ax.set_ylabel('Score')
ax.set_title('Model Performance on Test Set')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.xticks(rotation=45)

# Add the score values as annotations on the bars
for rect in rects:
    height = rect.get_height()
    ax.annotate('{:.4f}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Adjust the layout and show the plot
fig.tight_layout()
plt.show()

#In[19] What features are most important when the model is making predictions on the test set?

import eli5
from eli5.sklearn import PermutationImportance

# Create a scorer object for F1 with predicted probabilities
f1_scorer = make_scorer(f1_score, average='macro', zero_division=1)

# Compute feature importances with permutation importance
perm = PermutationImportance(xgb, random_state=42, scoring=f1_scorer).fit(X_test, y_test)

# Generate the bar chart of feature importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(X_test.columns, perm.feature_importances_, color='blue')

# Add chart labels, title, and legend
ax.set_xticklabels(X_test.columns, rotation=90, fontsize=12)
ax.set_ylabel("Feature Importance", fontsize=12)
ax.set_xlabel("Feature", fontsize=12)
ax.set_title("Feature Importance Chart", fontsize=14)
ax.legend(["Permutation Importance"], fontsize=12)
plt.show()
#In[20] Get predictions that are both correct and incorrect, uncertain and certain.
y_prob = xgb.predict_proba(X_test)

# find a correct prediction where the model is uncertain
uncertain_correct_mask = (y_pred == y_test) & (y_prob.max(axis=1) < 0.9)
uncertain_correct_index = uncertain_correct_mask.argmax()
uc_example = X_test.iloc[uncertain_correct_index]
actual_class_name = classes[y_test[uncertain_correct_index]]
predicted_class_name = classes[y_pred[uncertain_correct_index]]
print(f'Uncertain and Correct\nActual class: {actual_class_name}, Predicted class: {predicted_class_name}')

# find a correct prediction where the model is certain
certain_correct_mask = (y_pred == y_test) & (y_prob.max(axis=1) > 0.9)
certain_correct_index = certain_correct_mask.argmax()
cc_example = X_test.iloc[certain_correct_index]
actual_class_name = classes[y_test[certain_correct_index]]
predicted_class_name = classes[y_pred[certain_correct_index]]
print(f'\nCertain and Correct\nActual class: {actual_class_name}, Predicted class: {predicted_class_name}')

# find an incorrect prediction where the model is uncertain
uncertain_incorrect_mask = (y_pred != y_test) & (y_prob.max(axis=1) < 0.9)
uncertain_incorrect_index = uncertain_incorrect_mask.argmax()
ui_example = X_test.iloc[uncertain_incorrect_index]
actual_class_name = classes[y_test[uncertain_incorrect_index]]
predicted_class_name = classes[y_pred[uncertain_incorrect_index]]
print(f'\nUncertain and Incorrect\nActual class: {actual_class_name}, Predicted class: {predicted_class_name}')

# find an incorrect prediction where the model is certain
certain_incorrect_mask = (y_pred != y_test) & (y_prob.max(axis=1) > 0.9)
certain_incorrect_index = certain_incorrect_mask.argmax()
ci_example = X_test.iloc[certain_incorrect_index]
actual_class_name = classes[y_test[certain_incorrect_index]]
predicted_class_name = classes[y_pred[certain_incorrect_index]]
print(f'\nCertain and Incorrect\nActual class: {actual_class_name}, Predicted class: {predicted_class_name}')

#In[21] How is the model making predictions on individual samples from the test set?
#correct prediction where the model is uncertain
eli5.explain_prediction_xgboost(xgb, uc_example, feature_names=X_test.columns.tolist(), target_names=classes)

#In[22] How is the model making predictions on individual samples from the test set?
#correct prediction where the model is certain
eli5.explain_prediction_xgboost(xgb, cc_example, feature_names=X_test.columns.tolist(), target_names=classes)

#In[23] How is the model making predictions on individual samples from the test set?
#incorrect prediction where the model is uncertain
eli5.explain_prediction_xgboost(xgb, ui_example, feature_names=X_test.columns.tolist(), target_names=classes)

#In[24] How is the model making predictions on individual samples from the test set?
#incorrect prediction where the model is certain
eli5.explain_prediction_xgboost(xgb, ci_example, feature_names=X_test.columns.tolist(), target_names=classes)
# %%

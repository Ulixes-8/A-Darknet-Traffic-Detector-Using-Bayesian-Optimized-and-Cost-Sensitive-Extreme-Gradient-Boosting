
  # Set Hyperparameters: 
  # Hidden Layer activation: ReLU (torch.relu in PyTorch)
  # Output activation:	Softmax (torch.softmax in PyTorch)
  # Loss function: Cross entropy (torch.nn.CrossEntropyLoss in PyTorch)
  # Optimizer: adam (torch.optim.Adam in PyTorch)
  # """

# ---------------------------------- PYTORCH CLASSIFIER ---------------------------------------------------------------

#Workflow
#1 Get the data prepared for the model by turning it into tensors
#2 Build the model. Pick loss function and optimizer. Build a training loop.
#3 Fit the model to the data and make predictions.
#4 Evaluate the model and tune hyperparameters.
            
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

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

# Almost everything in PyTorch is called a "Module" (you build neural networks by stacking together Modules)
# Import PyTorch Dataset (you can store your data here) and DataLoader (you can load your data here)
from torch.utils.data import Dataset, DataLoader

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

#Only use top features
top_features = ['Dst_Port', 'Idle_Max', 'Src_Port', 'Fwd_IAT_Max', 'Fwd_IAT_Total', 'Idle_Mean', 'Flow_IAT_Max', 'Idle_Min', 'Fwd_IAT_Mean', 'Fwd_Header_Length', 'FWD_Init_Win_Bytes', 'Packet_Length_Min', 'Fwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean', 'Subflow_Bwd_Bytes', 'Flow_IAT_Mean', 'Bwd_Init_Win_Bytes', 'Fwd_Packets_per_second', 'Bwd_Segment_Size_Avg', 'Flow_Duration', 'Fwd_Seg_Size_Min', 'Flow_Packets_per_second', 'Flow_IAT_Min', 'Bwd_Packets_per_second', 'Average_Packet_Size', 'Flow_Bytes_per_second', 'Packet_Length_Mean', 'Packet_Length_Max', 'Bwd_Packet_Length_Max', 'Fwd_Packet_Length_Max', 'Total_Length_of_Fwd_Packet', 'Fwd_Segment_Size_Avg', 'Bwd_Packet_Length_Min', 'Packet_Length_Std', 'Fwd_IAT_Min', 'Total_Length_of_Bwd_Packet', 'Packet_Length_Variance', 'Fwd_Packet_Length_Mean', 'Bwd_Header_Length']
X = X[top_features]
X_normalized = sc.fit_transform(X)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#convert X_normalized to a numpy array
X = np.array(X_normalized)
y = np.array(y)

num_classes = len(np.unique(y))

# Compute the class weights
class_counts = np.bincount(y)
class_weights = 1. / class_counts
class_weights_normalized = class_weights / np.sum(class_weights)

#Convert to tensors 

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.int64)
class_weights_normalized = torch.from_numpy(class_weights_normalized).type(torch.float)

classes = np.unique(y)

#Create a bar chart that shows the distribution of the classes
plt.figure(figsize=(10, 5))
sns.countplot(y)
plt.title("Distribution of the classes")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()
        

# Split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=42) # make the random split reproducible


#print len(X_train), len(X_test), len(y_train), len(y_test)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)



# classes = ['Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'Streaming', 'VoIP']

#Classes are every unique value in the target column



# 2. Building a model ON https://www.learnpytorch.io/02_pytorch_classification/
# 1. Construct a model class that subclasses nn.Module

  # Hidden Layer activation: ReLU (torch.relu in PyTorch)
  # Output activation:	Softmax (torch.softmax in PyTorch)
  # Loss function: Cross entropy (torch.nn.CrossEntropyLoss in PyTorch)
  # Optimizer: adam (torch.optim.Adam in PyTorch)
  

  
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(in_features=39, out_features=100) 
        self.layer_2 = nn.Linear(in_features=100, out_features=100) 
        self.layer_3 = nn.Linear(in_features=100, out_features=100) 
        self.layer_4 = nn .Linear(in_features=100, out_features=100) 
        self.layer_5 = nn.Linear(in_features=100, out_features=7) 

        
            
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        # 4. Perform the forward pass computation
        x = self.layer_1(x) #This is a hidden layer. Input layer not specified 
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.relu(x)
        x = self.layer_4(x)
        x = torch.relu(x)
        x = self.layer_5(x) 
        x = torch.softmax(x, dim=1)
        return x
    
        
# 4. Create an instance of the model and send it to target device
model = NeuralNet().to(device)

loss_function = nn.CrossEntropyLoss(weight=class_weights_normalized.to(device)) #Expects logits as inputs
# loss_function = nn.CrossEntropyLoss() #Expects logits as inputs

# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

#adam optimizer with adaptive learning rate
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

#5. Build Model Evaluation functions 
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

#6. Train the model in training loop 
# Forward pass -> Calculate loss -> (Optimizer) Zero Gradients -> Backward pass(backprop) -> (Gradient des step) Update weights
# The outputs of the model are the logits, which are the raw values before the activation function is applied.
# We convert the logits to probabilities using the softmax function.
# We go from probabilities to class labels using the argmax function. 

torch.manual_seed(42) # Set random seed for reproducibility
torch.cuda.manual_seed(42) # Set random seed for reproducibility

epochs = 2000
X_train, y_train = X_train.to(device), y_train.to(device) #Toss training data on CUDA device
X_test, y_test = X_test.to(device), y_test.to(device) #Toss testing data on CUDA device

for epoch in range(epochs):
    model.train() #Put model in training mode
    
    #1. Forward pass
    y_logits = model(X_train) #Raw model outputs are logits (not probabilities)
    # y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1) #Turn logits into probabilities and then class labels
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    #2. Calculate loss and performance metrics while training
    loss = loss_function(y_logits, y_train) #Calculate loss using cross entropy loss function that takes in logits
    acc = accuracy_fn(y_train, y_pred) #Calculate accuracy

    #3 Zero gradients
    optimizer.zero_grad()
    
    #4. Backward pass
    loss.backward()
    
    #5. Update weights with gradient descent
    optimizer.step()
    
    #6. Testing the model (final model evaluation)
    model.eval() #Put model in evaluation mode
    with torch.inference_mode():
        #1. Forward pass
        # test_logits = model(X_test).squeeze() #Raw model outputs are logits (not probabilities)
        # test_pred = torch.argmax(torch.softmax(test_logits, dim=1), dim=1) #Turn logits into probabilities and then class labels

        test_logits = model(X_test) #Raw model outputs are logits (not probabilities)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        
        #2. Calculate loss and performance metrics
        test_loss = loss_function(test_logits, y_test) #Calculate loss using cross entropy loss function that takes in logits
        test_acc = accuracy_fn(y_test, test_pred) #Calculate accuracy


        #3. Print metrics
    if epoch % 10 == 0:
        # print(f"EPOCH: {epoch} | LOSS: {loss.item():.5f} | ACC: {acc:.2f}% | REC: {rec:.2f}% | PREC: {prec:.2f}% | F1: {f1:.2f}%") 
        print(f"EPOCH: {epoch} | LOSS: {loss:.5f} | ACC: {acc:.2f}% | TEST LOSS: {test_loss:.5f} | TEST ACC: {test_acc:.2f}%")        

#7 Hyperparameter Tuning the Model  
# Add more hidden layers
# Add more hidden units
# Increase number of epochs
# Experiment with different optimizers
# Experiment with different activation functions
# Experiment with different loss functions
# Employ 'experiment tracking' -- change one variable at a time so you know what causes what impact on the model. 



# # ---------------------------------- SK LEARN CLASSIFIER ---------------------------------------------------------------
# #DIARY 
# #There were no missing values in the dataset. 
# #There were null values in the dataset. Index(['Flow_Bytes_per_second'], dtype='object')
# #There were infinite values in the dataset. Index(['Flow_Bytes_per_second', 'Flow_Packets_per_second'], dtype='object')
# #Removed 49 rows from the dataset because of null and infinite values
# # Drop the categorical variables because they bias the dataset. 
# # Collected top features (RFI, m = 10, n_est = 100)
# # Normalized
# # Only using features with importance > 0.01
# # Learning rate = adaptive
# from sklearn.model_selection import cross_validate
 
# import pandas as pd
# import numpy as np
# import os
# import seaborn as sns
# import matplotlib.pyplot as plt


# # Setup device-agnostic code 
# df = pd.read_csv('C:/Users/ulixe/OneDrive/Desktop/Research/Darknet.csv')

# #Remove null and infinite values from the dataset
# df = df.replace([np.inf, -np.inf], np.nan)
# df = df.dropna()

# #We are trying to determine the application given the benign traffic
# df = df[df.Label_A != 'Darknet']
# df = df.drop(['Label_A'], axis=1)
# X = df.drop(['Label_B'], axis=1)
# y = df.Label_B

# #Handle the categorical variables
# categorical_variables = X.select_dtypes(include=['object']).columns
# #drop the categorical variables 
# X = X.drop(X[categorical_variables], axis=1)
            
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()

# # X_normalized = sc.fit_transform(X)            
# # # Random forest importances
# # from sklearn.ensemble import RandomForestClassifier
# # rf = RandomForestClassifier(n_jobs=8, n_estimators=1000, max_depth=10, random_state=0)
# # rf.fit(X_normalized, y)
# # importances = rf.feature_importances_
# # indices = np.argsort(importances)[::-1]
# # top_features = []
# # for f in range(X_normalized.shape[1]):
# #     if importances[indices[f]] > 0.01:
# #         print("%d. feature %d (%f) %s" % (f + 1, indices[f], importances[indices[f]], X.columns[indices[f]]))
# #         top_features.append(X.columns[indices[f]])
# # print(top_features)

# #Only use top features
# top_features = ['Dst_Port', 'Idle_Max', 'Src_Port', 'Fwd_IAT_Max', 'Fwd_IAT_Total', 'Idle_Mean', 'Flow_IAT_Max', 'Idle_Min', 'Fwd_IAT_Mean', 'Fwd_Header_Length', 'FWD_Init_Win_Bytes', 'Packet_Length_Min', 'Fwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean', 'Subflow_Bwd_Bytes', 'Flow_IAT_Mean', 'Bwd_Init_Win_Bytes', 'Fwd_Packets_per_second', 'Bwd_Segment_Size_Avg', 'Flow_Duration', 'Fwd_Seg_Size_Min', 'Flow_Packets_per_second', 'Flow_IAT_Min', 'Bwd_Packets_per_second', 'Average_Packet_Size', 'Flow_Bytes_per_second', 'Packet_Length_Mean', 'Packet_Length_Max', 'Bwd_Packet_Length_Max', 'Fwd_Packet_Length_Max', 'Total_Length_of_Fwd_Packet', 'Fwd_Segment_Size_Avg', 'Bwd_Packet_Length_Min', 'Packet_Length_Std', 'Fwd_IAT_Min', 'Total_Length_of_Bwd_Packet', 'Packet_Length_Variance', 'Fwd_Packet_Length_Mean', 'Bwd_Header_Length']
# X = X[top_features]
# X_normalized = sc.fit_transform(X)

# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import cross_val_predict
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import StratifiedKFold

# #Use cross validate to train the model and get the accuracy score, precision score, recall score, and f1 score
# def plot_matrix(cm, classes, title):
#   ax = sns.heatmap(cm, cmap="Blues", annot=True, fmt ='g', xticklabels=classes, yticklabels=classes, cbar=False)
#   ax.set(title=title, xlabel="Predicted Label", ylabel="True Label")

# performance_metrics = ["test_f1_macro", "test_accuracy", "test_precision_macro", "test_recall_macro"]
# foldCount = StratifiedKFold(10, shuffle=True, random_state=1)
# cv_results_mlp = []
# cv_results_xgb = []

# #import MLPClassifier from sklearn
# from sklearn.neural_network import MLPClassifier
# #Create a MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(500, 250, 125, 60, 30, 15, 7), max_iter=1000, verbose=True, random_state=0, solver="adam"
#                     , learning_rate_init=0.001, activation="relu", learning_rate="adaptive", early_stopping=True)

# # mlp.fit(X_normalized, y) FIT and then pickle
# cv_results = cross_validate(mlp, X_normalized, y, cv=foldCount,
#                                         scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'], 
#                                         n_jobs=8,
#                                         verbose=True)
# cv_results_mlp.append(cv_results)

# # Confusion Matrix
# y_pred = cross_val_predict(mlp, X_normalized, y, cv=foldCount, n_jobs = 8, verbose = True)
# conf_mat = confusion_matrix(y, y_pred)
# cm = np.array(conf_mat)
# # Classes
# classes = ['Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'Streaming', 'VoIP']
# title = "Multi Layer Perceptron (Model 3 - Benign App Cat.) Confusion Matrix"
# plot_matrix(cm, classes, title)
# print(conf_mat)

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

# # #import XGBoost Classifier
# # from xgboost import XGBClassifier


# # #XGBoost Classifier
# # from sklearn.preprocessing import LabelEncoder
# # # create an instance of the LabelEncoder class
# # le = LabelEncoder()
# # # fit and transform the target variable
# # target = le.fit_transform(y)
# # # xgb = XGBClassifier(random_state=0, n_estimators=100, max_depth=5, learning_rate=0.1)
# # # xgb = XGBClassifier(random_state=0, n_estimators=100, max_depth=8, learning_rate=0.1)
# # # xgb = XGBClassifier(random_state=0, n_estimators=100, max_depth=10, learning_rate=0.1)
# # xgb = XGBClassifier(random_state=0, n_estimators=1000, max_depth=25, learning_rate=0.01)

# # cv_results = cross_validate(xgb, X_normalized, target, cv=foldCount, n_jobs=8, verbose=1,
# #                                         scoring=['f1_macro', 'accuracy', 'precision_macro', 'recall_macro'])

# # # Confusion Matrix
# # y_pred = cross_val_predict(xgb, X_normalized, target, cv=foldCount, n_jobs = 8, verbose = 1)
# # conf_mat = confusion_matrix(target, y_pred)
# # cm = np.array(conf_mat)
# # # Classes
# # classes = ['Browsing', 'Chat', 'Email', 'File Transfer', 'P2P', 'Streaming', 'VoIP']
# # title = "XGBoost Confusion Matrix (Model 3 - Benign App Cat.)"
# # plot_matrix(cm, classes, title)
# # print(conf_mat)

# # #PRINT PERFORMANCE METRICS
# # cv_results_xgb.append(cv_results)
# # for i in performance_metrics:
# #     print(i)
# #     for j in cv_results_xgb[0][i]:
# #         print(j)
    
# # #XGB box plot
# # data1 = cv_results_xgb[0]['test_f1_macro']
# # data2 = cv_results_xgb[0]['test_accuracy']
# # data3 = cv_results_xgb[0]['test_precision_macro']
# # data4 = cv_results_xgb[0]['test_recall_macro']
# # allData = [data1,data2,data3,data4]

# ################### LEAVE THE BELOW UNCOMMENTED. IT PRODUCES THE BOXPLOT.
# ################### ALL THAT NEEDS TO BE SPECIFIED IS THE PLT.TITLE

# ### PLOT GENERATOR 
# sns.set(style='whitegrid')
# fig, ax = plt.subplots(figsize=(8,6))
# allData = [data1,data2,data3,data4]
# g = sns.boxplot(data=allData, width=0.7)

# ###TITLE SETTING TEMPLATES (ONLY ONE SHOULD BE UNCOMMENTED AT A TIME.)
# # plt.title("Multi Layer Perceptron (Model 3 - Benign App Cat.)", fontsize=16)
# plt.title("XGBoost (Model 3 - Benign App Cat.)", fontsize=16)

# # X labels
# xvalues = ["test_f1_macro", "test_accuracy", "test_precision_macro", "test_recall_macro"]

# # x-labels
# plt.xticks(np.arange(4), xvalues)

# # setting y values
# # plt.yticks(plt.yticks(np.arange(0,1,.1)))
# plt.yticks(np.arange(0,1.1,.1))

# ### CHANGE ORDER #### ### CHANGE X coordinates ###, change median, change textstr

# # Set colors of box plots 
# palette= ['#B7C3D0','#B7C3D0','#B7C3D0','#B7C3D0','#FF6A6A']
# color_dict = dict(zip(xvalues, palette))
# for i in range(0,4):
#     mybox = g.artists[i]
#     mybox.set_facecolor(color_dict[xvalues[i]])
    
# # F-Measure
# median = round(data1.median(),1)
# textstr = r"$\tilde {x}$" + f" = {median}"
# g.text(-0.19, 1.1, textstr, fontsize=13,) #### delete bbox

# # Accuracy
# median = round(data2.median(),1)
# textstr = r"$\tilde {x}$" + f" = {median}"
# g.text(.81, 1.1, textstr, fontsize=13,)

# # Precision
# median = round(data3.median(),1)

# textstr = r"$\tilde {x}$" + f" = {median}"
# g.text(1.81, 1.1, textstr, fontsize=13,) 

# # Recall
# median = round(data4.median(),1)
# textstr = r"$\tilde {x}$" + f" = {median}"
# g.text(2.81, 1.1, textstr, fontsize=13)

# #Plot
# plt.tight_layout()
# plt.show()


#################################################################### FEATURE SELECTION OPTIMIZATION ##################################################################################################


from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# Train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), random_state=0, max_iter=1000, early_stopping=True, learning_rate = 'adaptive', learning_rate_init = 0.1, activation = 'relu', solver = 'adam', alpha = 0.0001, batch_size = 256, verbose = True)
mlp.fit(X_train, y_train)


# Calculate Permutation Importances
result = permutation_importance(mlp, X_train, y_train, n_repeats=10, random_state=0, n_jobs=-1)

importances = result.importances_mean
indices = np.argsort(importances)[::-1]
for i in indices:
    print(f"{X.columns[i]}: {importances[i]}")

top_features = []

for i in range(64):
    top_features.append(X.columns[indices[i]])

print(f"Top features before optimization: {top_features}")




###########TEST TOP FEATURES WITH XGBOOST###########
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, make_scorer
from xgboost import XGBClassifier
from sklearn.feature_selection import SequentialFeatureSelector

# Define XGBClassifier with desired parameters
xgb = XGBClassifier(verbosity=0, random_state=0, n_estimators=100, max_depth=10,
                    learning_rate=0.35, gpu_id=0, tree_method='gpu_hist',
                    predictor='gpu_predictor')

# Create a dictionary to map feature names to their indices
feature_indices = {name: idx for idx, name in enumerate(top_features)}

# Initialize the optimal feature set and score
optimal_features = None
max_recall = 0
best_selected_feature_indices = None

# Iterate through top_features and add them one by one
for num_features in range(1, len(top_features) + 1):
    selected_features = top_features[:num_features]
    selected_feature_indices = [feature_indices[f] for f in selected_features]

    # Evaluate the current feature set using the validation set
    xgb.fit(X_train[:, selected_feature_indices], y_train)
    y_pred = xgb.predict(X_val[:, selected_feature_indices])
    recall = recall_score(y_val, y_pred)

    print(f"Number of features: {num_features}, Recall: {recall:.4f}, Selected features: {selected_features}")

    # Update the optimal feature set and score
    if recall > max_recall:
        max_recall = recall
        optimal_features = selected_features
        best_selected_feature_indices = selected_feature_indices
    else:
        continue

print(f"Optimal feature set: {optimal_features}, Max recall: {max_recall:.4f}")


X_train = X_train[:, best_selected_feature_indices]
print(f"X_train shape after feature selection: {X_train.shape}")
X_val = X_val[:, best_selected_feature_indices]
print(f"X_val shape after feature selection: {X_val.shape}")

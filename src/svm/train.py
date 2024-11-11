# %%
#Imports

import pandas as pd
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

from sklearn.model_selection import train_test_split

# %%
#Load the processed data:

train_data = pd.read_csv('/home/spocklight/tmp_new/processed_train_data.csv')
train_labels = pd.read_csv('/home/spocklight/tmp_new/processed_train_labels.csv').values.ravel()

#Taking only 10K rows for training in order to make the algorith training quicker

train_data_sample, _, train_labels_sample, _ = train_test_split(train_data, train_labels, train_size=10000, random_state=42)

# %%
#Loading the model and applying grid_search and cross validation (too slow)

'''
param_grid = {
    'C': [1, 10],
    'gamma': [0.01, 0.03, 0.1],
    'kernel': ['rbf']
}
grid_search = GridSearchCV(SVC(),
                           param_grid,
                           cv=5,
                           return_train_score=True,
                           verbose=3,
                           n_jobs = -1)

grid_search.fit(train_data, train_labels)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best average score (Cross-Validation): {grid_search.best_score_}")
best_model = grid_search.best_estimator_
'''

# %%
#Loading the model and applying randomized_search and cross validation 

param_dist = {
    'C': np.logspace(-1, 1,5),
    'gamma': np.logspace(-3, 0, 4),
    'kernel': ['rbf']
}

svc = SVC()
 
random_search = RandomizedSearchCV(svc,
                                   param_distributions=param_dist,
                                   n_iter=5,
                                   cv=3,
                                   n_jobs=-1,
                                   verbose=3,
                                   random_state=42)

random_search.fit(train_data_sample, train_labels_sample)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_}")

best_model = random_search.best_estimator_

results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df[['param_C', 'param_gamma', 'mean_test_score', 'std_test_score', 'rank_test_score']]

print("\nDetailed results:")
print(results_df.sort_values(by="rank_test_score"))

# %%
#In case we want to see the evolution of the training process with a bar

'''
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.03, 0.1],
    'kernel': ['rbf']
}
param_combinations = list(ParameterGrid(param_grid))

results = []
for params in tqdm(param_combinations, desc="Grid Search Progress"):
    model = SVC(**params)
    scores = cross_val_score(model, train_data, train_labels, cv=5)
    mean_score = np.mean(scores)
    results.append((params, mean_score))

best_params, best_score = max(results, key=lambda x: x[1])
print(f"\nBest parameters: {best_params}")
print(f"Best average score (Cross-Validation): {best_score}")


# %%
#Showing the avg performance of every combination:

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df[['param_C', 'param_gamma', 'mean_test_score', 'std_test_score', 'rank_test_score']]
print("\nDetailed results:")
print(results_df.sort_values(by="rank_test_score"))

'''
# %%
#Saving the model

joblib.dump(best_model, '/home/spocklight/Git/Git/Digit-Recognizer/models/best_svm_model.pkl')

# %%

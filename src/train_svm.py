# %%
#Imports

import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid, cross_val_score
from tqdm import tqdm

# %%
#Load the proces data:

train_data = pd.read_csv('/home/spocklight/Git/Git/Digit-Recognizer/data/processed_train_data.csv')
train_labels = pd.read_csv('/home/spocklight/Git/Git/Digit-Recognizer/data/processed_train_labels.csv').values.ravel()

# %%
#Loading the model and applying grid_search and cross validation

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.03, 0.1],
    'kernel': ['rbf']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True, verbose=3)
grid_search.fit(train_data, train_labels)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best average score (Cross-Validation): {grid_search.best_score_}")
best_model = grid_search.best_estimator_

#Best for now:

# Fitting 5 folds for each of 9 candidates, totalling 45 fits
# [CV 1/5] END C=0.1, gamma=0.01, kernel=rbf;, score=(train=0.950, test=0.945) total time= 6.2min
# [CV 2/5] END C=0.1, gamma=0.01, kernel=rbf;, score=(train=0.950, test=0.943) total time= 5.1min
# [CV 3/5] END C=0.1, gamma=0.01, kernel=rbf;, score=(train=0.951, test=0.943) total time= 9.9min
# [CV 4/5] END C=0.1, gamma=0.01, kernel=rbf;, score=(train=0.950, test=0.947) total time= 9.9min
# [CV 5/5] END C=0.1, gamma=0.01, kernel=rbf;, score=(train=0.950, test=0.948) total time=10.7min
# [CV 1/5] END C=0.1, gamma=0.03, kernel=rbf;, score=(train=0.967, test=0.957) total time= 7.7min
# [CV 2/5] END C=0.1, gamma=0.03, kernel=rbf;, score=(train=0.967, test=0.958) total time=11.7min
# [CV 3/5] END C=0.1, gamma=0.03, kernel=rbf;, score=(train=0.968, test=0.956) total time= 5.2min
# [CV 4/5] END C=0.1, gamma=0.03, kernel=rbf;, score=(train=0.967, test=0.960) total time=49.8min
# [CV 5/5] END C=0.1, gamma=0.03, kernel=rbf;, score=(train=0.966, test=0.964) total time= 5.9min
# [CV 1/5] END C=0.1, gamma=0.1, kernel=rbf;, score=(train=0.657, test=0.590) total time=806.4min
# [CV 2/5] END C=0.1, gamma=0.1, kernel=rbf;, score=(train=0.663, test=0.595) total time=28.7min
# [CV 3/5] END C=0.1, gamma=0.1, kernel=rbf;, score=(train=0.651, test=0.587) total time=17.6min
# [CV 4/5] END C=0.1, gamma=0.1, kernel=rbf;, score=(train=0.652, test=0.584) total time=17.8min
# [CV 5/5] END C=0.1, gamma=0.1, kernel=rbf;, score=(train=0.657, test=0.595) total time=20.8min
# [CV 1/5] END C=1, gamma=0.01, kernel=rbf;, score=(train=0.984, test=0.971) total time= 2.6min
# [CV 2/5] END C=1, gamma=0.01, kernel=rbf;, score=(train=0.984, test=0.972) total time= 2.5min
# [CV 3/5] END C=1, gamma=0.01, kernel=rbf;, score=(train=0.985, test=0.969) total time= 2.6min
# [CV 4/5] END C=1, gamma=0.01, kernel=rbf;, score=(train=0.984, test=0.973) total time= 2.5min
# [CV 5/5] END C=1, gamma=0.01, kernel=rbf;, score=(train=0.984, test=0.973) total time= 2.5min
# [CV 1/5] END C=1, gamma=0.03, kernel=rbf;, score=(train=0.997, test=0.981) total time= 6.1min
# [CV 2/5] END C=1, gamma=0.03, kernel=rbf;, score=(train=0.997, test=0.983) total time= 8.5min
# [CV 3/5] END C=1, gamma=0.03, kernel=rbf;, score=(train=0.997, test=0.979) total time= 8.2min


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
'''

# %%
#Showing the avg performance of every combination:

results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df[['param_C', 'param_gamma', 'mean_test_score', 'std_test_score', 'rank_test_score']]
print("\nDetailed results:")
print(results_df.sort_values(by="rank_test_score"))

# %%
#Saving the model

joblib.dump(best_model, '/home/spocklight/Git/Git/Digit-Recognizer/models/best_svm_model.pkl')
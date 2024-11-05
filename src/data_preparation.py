# %%
#Imports:

import pandas as pd
import os
import numpy as np
import random
from matplotlib import pyplot as plt

# %%
#Read the train and test datasets

print(os.getcwd())

train = pd.read_csv("/home/spocklight/Git/Git/Digit-Recognizer/data/train.csv")
test = pd.read_csv("/home/spocklight/Git/Git/Digit-Recognizer/data/test.csv")

# %%
#Shape and column names

print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')

print(f'Train columns: {train.columns}')
print(f'Test shape: {test.columns}')

#Each row represents a number and we have 785 columns, 
#each one representing if a pixel is black o white

#Remark that the train dataset has a label column that the test does not have
# %%
#Printing randomly some numbes of the train dataset

#For that, we select 9 ramdom indexes and create the figure

random_indices = random.sample(range(len(train)), 9)
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

for i, ax in enumerate(axes.flat):

        row = train.loc[random_indices[i]]
        label = row[0]
        image = np.array(row[1:], dtype='float').reshape((28, 28))

        ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(f'Label: {label}', fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.show()

# %%

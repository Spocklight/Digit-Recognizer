# %%
#Imports and config:

import pandas as pd
import os
import numpy as np
import random
import seaborn as sns

from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# %%
#Read the train and test datasets

print(os.getcwd())

train = pd.read_csv("/home/spocklight/Git/Git/Digit-Recognizer/data/train.csv")
test = pd.read_csv("/home/spocklight/Git/Git/Digit-Recognizer/data/test.csv")

# %%
#First exploration

print(train.info())
print(train.describe())
print(test.info())
print(test.describe())

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
#Checking if there are nulls

def show_nulls(df, name_df):

    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    
    if nulls.empty:
        print(f"There are no null values in the {name_df} dataframe.")
    else:
        print(f"{name_df} rows with null values:\n{nulls}")

show_nulls(train, "Train")
show_nulls(test, "Test")

# %%
#Value counts of some pixels of train:

random_indices = random.sample(range(1, len(train)), 2)

for i in range(len(random_indices)):
    value_counts_train = train.iloc[random_indices[i]].value_counts()
    value_counts_train = value_counts_train[value_counts_train > 2]
    plt.figure(figsize=(10, 6))
    value_counts_train.plot(kind='bar', color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title(f'Value Counts of pixel {random_indices[i]} in the Training Dataset')
    plt.show()

# %%
#Checking the max value of each pixel

#train.select_dtypes(include=[np.number]).max().sort_values(ascending=False)

image = train.select_dtypes(include=[np.number]).max().values[1:].reshape(28, 28)

plt.figure(figsize=(8, 8))
sns.heatmap(image, cmap="YlGnBu", cbar=True, square=True)
plt.title("Max values of each pixel in the 28x28 train dataset")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.show()

# %%
#Distribution of the label column to check if the representation of each number

sns.countplot(x=train['label'])
plt.title('Distribución de etiquetas en el dataset de entrenamiento')
plt.show()

# %%
#Printing randomly some numbes of the train dataset
#For that, we select 9 ramdom indexes and create the figure

random_indices = random.sample(range(1, len(train)), 9)
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
#After some exploration, we normalize the pixels:

train_data = train.drop(columns=['label']) / 255.0
train_labels = train['label']
test_data = test / 255.0

# %%
#We could apply some dimension reduction usig PCA to improve the precision-performance balance
#We won't do it for now

'''
from sklearn.decomposition import PCA

# Reducir dimensionalidad (opcional)
pca = PCA(n_components=50)  # Ajusta n_components para optimizar el balance entre precisión y rendimiento
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)
'''

# %%
#Save processed data:

#Creamos previamente una carpeta temporal que se borre cada semana usando un cronjob:

#mkdir /home/spocklight/tmp_new
#crontab -e (y seleccionamos 1, para usar el editor nano)
#0 0 * * 0 rm -rf /home/spocklight/tmp_new (añadimos esta línea + ctrlX + Y)

train_data.to_csv('/home/spocklight/tmp_new/processed_train_data.csv', index=False)
train_labels.to_csv('/home/spocklight/tmp_new/processed_train_labels.csv', index=False)
test_data.to_csv('/home/spocklight/tmp_new/processed_test_data.csv', index=False)

# %%
#Techniques like data augmentation (creating slightly modified versions of the original images) can also improve generalization and performance.
#We apply them in the CNN algorithm.
# %%

# Google Girl Hackathon 2024 Project
# Submitted by Ayushi Aggarwal
# NIT Kurukshetra

import time
import math

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, \
    LSTM
import tensorflow as tf

# Matplotlib config
plt.rc('image', cmap='gray_r')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')

try:
    train = pd.read_csv('./Data/train.csv', nrows=4095 * 100,
                        dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

except FileNotFoundError:
    print("Please Download the data and save in Data folder for the best results!")
    train = pd.read_csv('./Data/sample.csv', nrows=4095 * 100,
                        dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

print('Data frame shape:{}'.format(train.shape))

# visualize 1% of samples data, every 100th datapoint

n_samples = 4096 * 25 + 1
n_step = 4096

train_ad_sample_df = train['acoustic_data'].values[:n_samples]
train_ttf_sample_df = train['time_to_failure'].values[:n_samples]


# function for plotting based on both features
def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df,
                      title="Acoustic data and time to failure: First n sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample_df, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)
    plt.show()


plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

# Free up the memory
del train_ad_sample_df
del train_ttf_sample_df

sequence_length = 4095
n_sequences = 40000
num_features = 1
BATCH_SIZE = 128

# If train.csv is not available, use sample data
try:
    X_train = pd.read_csv('./Data/train.csv', nrows=sequence_length * n_sequences,
                          usecols=['acoustic_data'], dtype={
            'acoustic_data': np.int16})  # .values.reshape(-1,4095,num_features) #Read values and reshape

    Y_train = pd.read_csv('./Data/train.csv', nrows=sequence_length * n_sequences,
                          usecols=['time_to_failure'], dtype={'time_to_failure': np.float64}).values.reshape(-1, 4095,
                                                                                                             1)
except:
    print("Please Download the data and save in Data folder for the best results!")
    X_train = pd.read_csv('./Data/sample.csv', nrows=sequence_length * n_sequences,
                          usecols=['acoustic_data'], dtype={
            'acoustic_data': np.int16})  # .values.reshape(-1,4095,num_features) #Read values and reshape

    Y_train = pd.read_csv('./Data/sample.csv', nrows=sequence_length * n_sequences,
                          usecols=['time_to_failure'], dtype={'time_to_failure': np.float64}).values.reshape(-1, 4095,
                                                                                                             1)

Y_train = np.float32(Y_train[:, -1, -1])  # We only need the last value at end of sequence
print(X_train.shape, Y_train.shape)

# Scale input data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train_scaled = np.float32(X_train_scaled.reshape(-1, 4095, num_features))
n_samples = X_train_scaled.shape[0]
print(X_train_scaled.shape, Y_train.shape)
print(X_train_scaled.dtype, Y_train.dtype)


# lets create a function to generate some statistical features based on the training data
def gen_features(X):
    strain = []
    strain.append(X.mean())
    strain.append(X.std())
    strain.append(X.min())
    strain.append(X.max())
    strain.append(X.kurtosis())
    strain.append(X.skew())
    strain.append(np.quantile(X, 0.01))
    strain.append(np.quantile(X, 0.05))
    strain.append(np.quantile(X, 0.95))
    strain.append(np.quantile(X, 0.99))
    strain.append(np.abs(X).max())
    strain.append(np.abs(X).mean())
    strain.append(np.abs(X).std())
    return pd.Series(strain)


# Training data for DNN

try:
    train = pd.read_csv('./Data/train.csv', iterator=True, chunksize=150000,
                        dtype={'acoustic_data': np.int16})
except:
    train = pd.read_csv('./Data/sample.csv', iterator=True, chunksize=150000,
                        dtype={'acoustic_data': np.int16})
num_statistical_features = 13
X_train_features = pd.DataFrame()
Y_train_features = pd.Series()
for df in train:
    features = gen_features(df['acoustic_data'])
    X_train_features = X_train_features._append(features, ignore_index=True)
    Y_train_features = Y_train_features._append(pd.Series(df['time_to_failure'].values[-1]))

# Scale input data
scaler = StandardScaler()
scaler.fit(X_train_features)
X_train_scaled = scaler.transform(X_train_features)
X_train_scaled = np.float32(X_train_scaled.reshape(-1, num_statistical_features))
Y_train = np.float32(Y_train_features.values)
n_samples = X_train_scaled.shape[0]
print(X_train_scaled.shape, Y_train.shape)
print(X_train_scaled.dtype, Y_train.dtype)


def get_training_dataset(batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, Y_train.reshape(-1, 1)))
    dataset = dataset.cache()  # this small dataset can be entirely cached in RAM, for TPU this is important to get good performance from such a small dataset
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)  # Shuffle, repeat, and batch the examples.
    dataset = dataset.repeat()  # Mandatory for Keras for now
    dataset = dataset.batch(batch_size,
                            drop_remainder=True)  # drop_remainder is important on TPU, batch size must be fixed
    dataset = dataset.prefetch(
        -1)  # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
    return dataset  # Return the dataset


# instantiate the datasets
training_dataset = get_training_dataset(BATCH_SIZE)

# For TPU, we will need a function that returns the dataset
training_input_fn = lambda: get_training_dataset(BATCH_SIZE)


def create_earthquake_MLP_model():
    earthquake_MLP_model = Sequential()  # Initialising the ANN
    earthquake_MLP_model.add(BatchNormalization(input_shape=(num_statistical_features,)))
    earthquake_MLP_model.add(Dense(units=20, activation='relu'))  # , input_dim = 13
    earthquake_MLP_model.add(BatchNormalization())
    earthquake_MLP_model.add(Dense(units=20, activation='relu'))
    earthquake_MLP_model.add(Dense(units=20, activation='relu'))
    earthquake_MLP_model.add(Dense(units=20, activation='relu'))
    earthquake_MLP_model.add(Dense(units=20, activation='relu'))
    earthquake_MLP_model.add(Dense(units=20, activation='relu'))

    earthquake_MLP_model.add(Dense(units=10, activation='relu'))
    earthquake_MLP_model.add(Dense(units=10, activation='relu'))
    earthquake_MLP_model.add(Dense(units=10, activation='relu'))
    earthquake_MLP_model.add(Dense(units=10, activation='relu'))
    earthquake_MLP_model.add(Dense(units=10, activation='relu'))
    earthquake_MLP_model.add(Dense(units=1, activation='linear'))
    earthquake_MLP_model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    earthquake_MLP_model.summary()

    return earthquake_MLP_model


DECAY = False
# set up learning rate decay
if DECAY:
    lr_decay = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0001 + 0.02 * math.pow(0.5, 1 + epoch),
                                                        verbose=True)
else:
    lr_decay = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001, verbose=True)

trained_model = create_earthquake_MLP_model()

EPOCHS = 500

# Uncomment Below line if using sample dataset
# n_samples = 4100
steps_per_epoch = n_samples // BATCH_SIZE  # 60,000 items in this dataset
print(f'Iterations per epoch:{steps_per_epoch}')

# To show time elapsed
start_time = time.time()

history = trained_model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS)

print(f'Time elapsed:{(time.time() - start_time) / 60.0} minutes')

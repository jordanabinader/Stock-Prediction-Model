import pandas as pd
import matplotlib.pyplot as plt
from utils.supervised import create_windowed_dataframe 
from utils.windowed_df_to_date_X_y import windowed_df_to_date_X_y

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras import layers
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from copy import deepcopy
from utils.plot_util import plot_all_predictions

# Load data from CSV
df = pd.read_csv('AAPL.csv')

# Calculate Returns
df['Returns'] = (df['Close'] - df['Open']) / df['Open']

# Select relevant columns
df = df[['Date', 'Returns']]

# Convert 'Date' column to datetime format and set it as index
df['Date'] = pd.to_datetime(df['Date'])
df.index = df.pop('Date')

# Create windowed dataframe
windowed_df = create_windowed_dataframe(df, '2010-01-13', '2023-09-07')

# Convert windowed dataframe to date, features, and target arrays
dates, X, y = windowed_df_to_date_X_y(windowed_df)

# Split data into training, validation, and test sets
q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

# Step 1: Standardize the 'Returns' column
scaler = StandardScaler()
X_train[:, :, 0] = scaler.fit_transform(X_train[:, :, 0])
X_val[:, :, 0] = scaler.transform(X_val[:, :, 0])
X_test[:, :, 0] = scaler.transform(X_test[:, :, 0])

# Step 2: Fit a Gaussian Mixture Model (GMM) to the 'Returns' column
n_components = 2 
gmm = GaussianMixture(n_components=n_components, random_state=0)
gmm.fit(X_train[:, :, 0].reshape(-1, 1))

model = Sequential([
    layers.Input((3, 1)),
    layers.LSTM(64),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
# Compile the model, loss: Mean-squared error, learning_rate - trial and error
model.compile(
    loss='mse', 
    optimizer=Adam(learning_rate=0.001),
    metrics=['mean_absolute_error']
)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500)

# Make predictions
train_predictions = model.predict(X_train).flatten()
val_predictions = model.predict(X_val).flatten()
test_predictions = model.predict(X_test).flatten()

# Recursive predictions
recursive_predictions = []
recursive_dates = np.concatenate([dates_val, dates_test])

for target_date in recursive_dates:
    # Copy the last window from the training data
    last_window = deepcopy(X_train[-1])
    
    # Predict the next value based on the last window
    next_prediction = model.predict(np.array([last_window])).flatten()
    
    # Append the prediction to the list
    recursive_predictions.append(next_prediction)
    
    # Update the last window with the new prediction
    last_window[:-1] = last_window[1:]
    last_window[-1] = next_prediction

plot_all_predictions(dates_train, train_predictions,
                         dates_val, val_predictions,
                         dates_test, test_predictions,
                         recursive_dates, recursive_predictions,
                         y_train, y_val, y_test)
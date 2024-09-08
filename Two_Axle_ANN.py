import numpy as np
import pandas as pd
import tensorflow as tf

## Data Preprocessing
# Importing the dataset
dataset = pd.read_excel('Two_Axle_Sim_Data.xlsx')
X = dataset.iloc[2:, :12].values
Y = dataset.iloc[2:, [16,20,22,25,26,29]].values
# Y = dataset.iloc[2:, [16,20,22,23,25,26,27,29]].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

X_train = np.array(X_train, dtype=np.float32)
Y_train = np.array(Y_train, dtype=np.float32)
print("X_train max:", np.max(X_train))
print("X_train min:", np.min(X_train))
print("Y_train max:", np.max(Y_train))
print("Y_train min:", np.min(Y_train))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scx = StandardScaler()
scy = StandardScaler()
X_train = scx.fit_transform(X_train)
X_test = scx.transform(X_test)
# Y_train = scy.fit_transform(Y_train)
# Y_test = scy.transform(Y_test)

import joblib
joblib.dump(scx, 'scaler.pkl')

## Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding layers
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=Y.shape[1]))

## Training the ANN
# Compiling the ANN
ann.compile(optimizer='adam', loss='mean_absolute_percentage_error')

# Training the ANN on the Training set
ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)

## Making the predictions and evaluating the model
# Predicting the Test set results
Y_pred = ann.predict(X_test)

# Create Multi-Index DataFrame for predicted and actual values
num_samples = Y_pred.shape[0]  # Assuming same number of samples for predicted and actual
num_features = Y_pred.shape[1]  # Assuming same feature dimensions

# Flatten both arrays for better comparison
Y_pred_flattened = Y_pred.reshape(num_samples, num_features)
Y_test_flattened = Y_test.reshape(num_samples, num_features)

# Create DataFrame
result_df = pd.DataFrame({
    ('Predicted', 'Feature' + str(i)): Y_pred_flattened[:, i] for i in range(num_features)
})
result_df = result_df.join(pd.DataFrame({
    ('Actual', 'Feature' + str(i)): Y_test_flattened[:, i] for i in range(num_features)
}))

# Display the DataFrame
print(result_df)

## Evaluating the Model
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# Average Mean Squared Error
mape_avg  = mean_absolute_percentage_error(Y_test, Y_pred, multioutput='uniform_average')
print(f'Average Mean absolute Error: {mape_avg }')

# R-squared Score
r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')
print(f'R-squared score: {r2}')

# Saving the model to a file
ann.save('two_axle_ann_model.h5')
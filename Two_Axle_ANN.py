import numpy as np
import pandas as pd
import tensorflow as tf
tf.config.run_functions_eagerly(True)


## Partial Dense
class PartialDense(tf.keras.layers.Layer):
    def __init__(self, units, connections_per_node, **kwargs):
        super(PartialDense, self).__init__(**kwargs)
        self.units                = units
        self.connections_per_node = connections_per_node

    def build(self, input_shape):
        self.input_dim = input_shape[-1]  # Automatically get the input dimension
        self.kernel    = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.units),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.bias      = self.add_weight(name='bias',
                                      shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True)

        # Initialize sparse connection matrix as all zeros
        self.connection_matrix = tf.zeros((self.input_dim, self.units), dtype=tf.float32)

        # Randomly select input node indices to be connected to each output node
        selected_indices = tf.random.shuffle(tf.range(self.input_dim))[:self.connections_per_node]
        
        # Create the indices matrix, representing [input node index, output node index]
        selected_indices = tf.tile(tf.expand_dims(selected_indices, axis=-1), [1, self.units])
        output_indices   = tf.tile(tf.range(self.units)[tf.newaxis, :], [self.connections_per_node, 1])

        indices = tf.stack([tf.reshape(selected_indices, [-1]), tf.reshape(output_indices, [-1])], axis=1)

        # Create the updates matrix
        updates = tf.ones([self.connections_per_node * self.units], dtype=tf.float32)

        # Use tensor_scatter_nd_update to update the connection matrix
        self.connection_matrix = tf.tensor_scatter_nd_update(self.connection_matrix, indices, updates)

    def call(self, inputs):
        # Retain only the weights of partially connected nodes
        output = tf.matmul(inputs, self.kernel * self.connection_matrix) + self.bias
        return output


## Data Preprocessing
# Importing the dataset
dataset = pd.read_excel('Two_Axle_Sim_Data.xlsx')
X = dataset.iloc[2:, :12].values
# Y = dataset.iloc[2:, [16,20,22,25,26,29]].values
Y = dataset.iloc[2:, [16,20,22,23,25,26,27,29]].values

# Units: N -> kN
Y[:, 3] /= 1000   
Y[:, 6] /= 1000

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
X_train = scx.fit_transform(X_train)
X_test  = scx.transform(X_test)
# scy = StandardScaler()
# Y_train = scy.fit_transform(Y_train)
# Y_test  = scy.transform(Y_test)

import joblib
joblib.dump(scx, 'scaler.pkl')


## Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding layers
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=64, activation='relu'))
ann.add(tf.keras.layers.Dense(units=128, activation='relu'))

ann.add(PartialDense(units=Y.shape[1], connections_per_node=32))

ann.add(tf.keras.layers.Dense(units=Y.shape[1]))


## Training the ANN
# Compiling the ANN
ann.compile(optimizer='adam', loss='mean_absolute_percentage_error')

# Training the ANN on the Training set
ann.fit(X_train, Y_train, batch_size = 32, epochs = 30)


## Making the predictions and evaluating the model
# Predicting the Test set results
Y_pred = ann.predict(X_test)

# Create Multi-Index DataFrame for predicted and actual values
num_samples  = Y_pred.shape[0]  # Assuming same number of samples for predicted and actual
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
print(f'Average Mean absolute percentage Error: {mape_avg }')

# R-squared Score
r2 = r2_score(Y_test, Y_pred, multioutput='uniform_average')
print(f'R-squared score: {r2}')

# Saving the model to a file
ann.save('two_axle_ann_model.h5')
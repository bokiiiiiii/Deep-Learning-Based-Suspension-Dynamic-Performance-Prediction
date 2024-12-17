import numpy as np
import tensorflow as tf
from scipy.optimize import minimize


## Partial Dense
class PartialDense(tf.keras.layers.Layer):
    def __init__(self, units, connections_per_node, **kwargs):
        super(PartialDense, self).__init__(**kwargs)
        self.units = units
        self.connections_per_node = connections_per_node

    def build(self, input_shape):
        self.input_dim = input_shape[-1]  # Automatically get the input dimension
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units,), initializer="zeros", trainable=True
        )

        # Initialize sparse connection matrix as all zeros
        self.connection_matrix = tf.zeros(
            (self.input_dim, self.units), dtype=tf.float32
        )

        # Randomly select input node indices to be connected to each output node
        selected_indices = tf.random.shuffle(tf.range(self.input_dim))[
            : self.connections_per_node
        ]

        # Create the indices matrix, representing [input node index, output node index]
        selected_indices = tf.tile(
            tf.expand_dims(selected_indices, axis=-1), [1, self.units]
        )
        output_indices = tf.tile(
            tf.range(self.units)[tf.newaxis, :], [self.connections_per_node, 1]
        )

        indices = tf.stack(
            [tf.reshape(selected_indices, [-1]), tf.reshape(output_indices, [-1])],
            axis=1,
        )

        # Create the updates matrix
        updates = tf.ones([self.connections_per_node * self.units], dtype=tf.float32)

        # Use tensor_scatter_nd_update to update the connection matrix
        self.connection_matrix = tf.tensor_scatter_nd_update(
            self.connection_matrix, indices, updates
        )

    def call(self, inputs):
        # Retain only the weights of partially connected nodes
        output = tf.matmul(inputs, self.kernel * self.connection_matrix) + self.bias
        return output


import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

model = tf.keras.models.load_model(
    "three_axle_ann_model.h5", custom_objects={"PartialDense": PartialDense}
)


def objective(x_input):

    x_input = np.array(x_input, dtype=np.float32).reshape(1, -1)
    x_input = scx.transform(x_input)

    y_pred = model.predict(x_input, verbose=0)

    return y_pred[0, 9]


bounds = [
    (20337.8 / 2 * 2 / 4, 20337.8 / 2 * 2 / 4),
    (458.4, 458.4),
    (562239.5791 / 2 * 2 / 4, 562239.5791 / 2 * 2 / 4),
    (11972.71197 * 0.6, 11972.71197 * 1.4),
    (76014.16394 * 0.6, 76014.16394 * 1.4),
    (548254.4557 * 0.6, 548254.4557 * 1.4),
    (1.245792544 * 0.6, 1.245792544 * 1.4),
    (1.024981218 * 0.6, 1.024981218 * 1.4),
    (0.623897077 * 0.6, 0.623897077 * 1.4),
    (0.592907552 * 0.6, 0.592907552 * 1.4),
    (55.25703168 * 0.6, 55.25703168 * 1.4),
    (2.658320979 * 0.6, 2.658320979 * 1.4),
]

import joblib

scx = joblib.load("scaler.pkl")
x0 = scx.transform(
    [
        [
            20337.8 / 2 * 2 / 4,
            458.4,
            562239.5791 / 2 * 2 / 4,
            11972.71197,
            76014.16394,
            548254.4557,
            1.245792544,
            1.024981218,
            0.623897077,
            0.592907552,
            55.25703168,
            2.658320979,
        ]
    ]
)[0].tolist()

result = minimize(
    fun=objective,
    x0=x0,
    bounds=bounds,
    method="L-BFGS-B",
    options={"disp": True},
)

optimized_x = result.x
minimized_raw33 = result.fun

print("Optimized Input (x):", optimized_x)
print("Minimized raw33 Value:", minimized_raw33)

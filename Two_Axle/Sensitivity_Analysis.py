import numpy as np
import pandas as pd
import tensorflow as tf
import random

tf.config.run_functions_eagerly(True)


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


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def perturbation_sensitivity_analysis(model, X_test, Y_test, perturbation=0.005):
    """
    Perform sensitivity analysis by perturbing each input feature and measuring the effect on outputs.

    Args:
        model: Trained model for prediction.
        X_test: Original test data inputs.
        Y_test: Ground truth outputs for comparison.
        perturbation: Fractional change to apply to input features for sensitivity analysis.

    Returns:
        sensitivity_df: DataFrame showing the sensitivity of each output to each input.
    """
    # Number of inputs and outputs
    num_inputs = X_test.shape[1]
    num_outputs = Y_test.shape[1]

    # Get the original predictions
    original_predictions = model.predict(X_test)

    # Store sensitivities
    sensitivities = np.zeros((num_inputs, num_outputs))

    # Loop through each input feature and perturb it
    for i in range(num_inputs):
        X_test_perturbed = X_test.copy()
        # Apply a small perturbation to the i-th feature
        X_test_perturbed[:, i] += X_test[:, i] * perturbation

        # Get the new predictions
        perturbed_predictions = model.predict(X_test_perturbed)

        # Calculate the sensitivity for each output
        sensitivity = np.mean(
            np.abs(
                (perturbed_predictions - original_predictions) / original_predictions
            )
            * 100,
            axis=0,
        )
        # sensitivity = np.mean(np.abs(perturbed_predictions - original_predictions), axis=0)
        sensitivities[i, :] = sensitivity

    # Create a DataFrame to display the results
    sensitivity_df = pd.DataFrame(
        sensitivities,
        index=[
            "ms",
            "mus",
            "iy",
            "cs",
            "ks",
            "kt",
            "h",
            "kratio",
            "cratio",
            "wr",
            "cg",
            "wb",
        ],
        columns=[
            "sws1_max",
            "sws2_max",
            "a_rms",
            "dtl1_rms",
            "theta_rms",
            "dtheta_rms",
            "dtl2_rms",
            "SDPI",
        ],
    )

    return sensitivity_df


if __name__ == "__main__":

    set_random_seed(22)

    ## Data Preprocessing
    # Importing the dataset
    dataset = pd.read_excel("Two_Axle_Sim_Data.xlsx")
    X = dataset.iloc[2:, :12].values
    Y = dataset.iloc[2:, [16, 20, 22, 23, 25, 26, 27, 29]].values

    # Units: N -> kN
    Y[:, 3] /= 1000
    Y[:, 6] /= 1000

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0
    )

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
    X_test = scx.transform(X_test)

    # Perform the sensitivity analysis
    ann = tf.keras.models.load_model(
        "two_axle_ann_model.h5", custom_objects={"PartialDense": PartialDense}
    )
    sensitivity_df = perturbation_sensitivity_analysis(ann, X_train, Y_train)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sensitivity_matrix = sensitivity_df.values

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 100,
        }
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sensitivity_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=sensitivity_df.columns,
        yticklabels=sensitivity_df.index,
        cbar_kws={"label": "Sensitivity"},
        linewidths=0.5,
        linecolor="gray",
    )

    plt.title("Sensitivity Analysis Heatmap", pad=20, fontsize=14)
    plt.xlabel("Outputs", labelpad=10)
    plt.ylabel("Inputs", labelpad=10)

    plt.tight_layout()

    # plt.savefig('sensitivity_analysis_heatmap.png', dpi=300, bbox_inches='tight')

    plt.show()

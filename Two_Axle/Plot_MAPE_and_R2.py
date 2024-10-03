import json
import matplotlib.pyplot as plt
import numpy as np

# Apply smoothing with a window size
WINDOW_SIZE = 25

# Load data from JSON files
with open("two_axle_ann_model_history.json", "r") as f:
    loaded_ann_history = json.load(f)

with open("two_axle_ann_model_epoch.json", "r") as f:
    loaded_ann_epoch = json.load(f)

with open("two_axle_ann_mln_model_history.json", "r") as f:
    loaded_ann_mln_history = json.load(f)

with open("two_axle_ann_mln_model_epoch.json", "r") as f:
    loaded_ann_mln_epoch = json.load(f)

# Extract MAPE and R² from the training history
ann_mape = np.array(loaded_ann_history["mean_absolute_percentage_error"])
ann_mln_mape = np.array(loaded_ann_mln_history["mean_absolute_percentage_error"])
ann_r2_scores = np.array(loaded_ann_history["r2_score"])
ann_mln_r2_scores = np.array(loaded_ann_mln_history["r2_score"])


# Function to apply simple moving average for smoothing
def smooth_data(data, WINDOW_SIZE):
    return np.convolve(data, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode="valid")


ann_mape_smooth = smooth_data(ann_mape, WINDOW_SIZE)
ann_mln_mape_smooth = smooth_data(ann_mln_mape, WINDOW_SIZE)
ann_r2_scores_smooth = smooth_data(ann_r2_scores, WINDOW_SIZE)
ann_mln_r2_scores_smooth = smooth_data(ann_mln_r2_scores, WINDOW_SIZE)

# Adjust the epoch data length for the smoothed data
smoothed_epochs_ann = loaded_ann_epoch[: len(ann_mape_smooth)]
smoothed_epochs_mln = loaded_ann_mln_epoch[: len(ann_mln_mape_smooth)]

# Set the style for arXiv paper - Black, White, Gray, and Times New Roman font
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"

# ---- Plot MAPE ----
fig, ax1 = plt.subplots(figsize=(7, 6))  # MAPE plot
ax1.plot(
    smoothed_epochs_ann,
    ann_mape_smooth,
    color="black",
    label="DNN Model",
    linestyle="-",
)
ax1.plot(
    smoothed_epochs_mln,
    ann_mln_mape_smooth,
    color="gray",
    label="MTL-DBN-DNN Model",
    linestyle="--",
)
ax1.set_yscale("log")  # Optional log scale if needed

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend()
ax1.set_xlabel("Training Epoch", fontsize=12)
ax1.set_ylabel("Mean Absolute Percentage Error", fontsize=12)
ax1.set_title(
    "Convergence of Mean Absolute Percentage Error Across Training Epochs",
    fontsize=14,
    fontweight="bold",
)
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()

# ---- Plot R² ----
fig, ax2 = plt.subplots(figsize=(7, 6))  # R² plot
ax2.plot(
    smoothed_epochs_ann,
    ann_r2_scores_smooth,
    color="black",
    label="DNN Model",
    linestyle="-",
)
ax2.plot(
    smoothed_epochs_mln,
    ann_mln_r2_scores_smooth,
    color="gray",
    label="MTL-DBN-DNN Model",
    linestyle="--",
)

# Labeling the plot for R²
ax2.set_xlabel("Training Epoch", fontsize=12)
ax2.set_ylabel("R-squared", fontsize=12)
ax2.set_title("R-squared Across Training Epochs", fontsize=14, fontweight="bold")
ax2.legend()
ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()

# Display the plots
plt.show()

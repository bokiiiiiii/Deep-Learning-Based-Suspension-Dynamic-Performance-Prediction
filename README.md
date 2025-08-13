# Deep Learning Based Suspension Dynamic Performance Prediection

Lin, B., and Lin, K., "Deep Learning–Based Prediction of Suspension Dynamics Performance in Multi-Axle Vehicles," SAE Int. J. Passeng. Veh. Syst. 18(3), 2025. [paper](https://doi.org/10.4271/15-18-03-0014)

## Abstract
This paper is mainly to present a deep learning-based framework for predicting the dynamic performance of suspension systems for multi-axle vehicles which emphasizes the integration of machine learning with traditional vehicle dynamics modeling. A Multi-Task Deep Belief Network Deep Neural Network (MTL-DBN-DNN) was developed to capture the relationships between key vehicle parameters and suspension performance. The model was trained on the data generated from numerical simulations. This model not only demonstrated superior prediction accuracy but also increased computational speed compared to traditional DNN models. A comprehensive sensitivity analysis was conducted to understand the impact of various vehicle and suspension parameters on dynamic suspension performance. Furthermore, we introduce the Suspension Dynamic Performance Index (SDPI) in order to measure and quantify overall suspension performance and the effectiveness of multiple parameters. The findings highlight the effectiveness of multitask learning in improving predictive models for complex vehicle systems.

## Project Structure

This repository is organized as follows:

-   `README.md`: This file, providing an overview of the project.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
-   `Three_Axle/`: Contains code and data related to the three-axle vehicle model.
    -   `class_DE_100m.mat`: MATLAB data file.
    -   `Optimization.py`: Python script for optimization tasks.
    -   `scaler.pkl`: Pickled scaler object for data normalization.
    -   `three_axle_ann_model.h5`: Trained  model for the three-axle vehicle.
    -   `Three_Axle_ANN.py`: Python script for the Artificial Neural Network related to the three-axle vehicle.
    -   `Three_Axle_main.m`: Main MATLAB script for the three-axle vehicle simulation.
    -   `Three_Axle_Sim_Data.xlsx`: Excel file containing simulation data for the three-axle vehicle.
    -   `Three_Axle_Simulation_ISO8608.m`: MATLAB script for simulation according to ISO8608 standard for the three-axle vehicle.
-   `Two_Axle/`: Contains code and data related to the two-axle vehicle model.
    -   `class_DE_100m.mat`: MATLAB data file.
    -   `Plot_MAPE_and_R2.py`: Python script for plotting Mean Absolute Percentage Error (MAPE) and R-squared (R2) metrics.
    -   `Sensitivity_Analysis.py`: Python script for performing sensitivity analysis.
    -   `Two_Axle_ANN_DBN.py`: Python script for the Deep Belief Network (DBN) based ANN for the two-axle vehicle.
    -   `two_axle_ann_mln_model_epoch.json`: JSON file storing epoch data for the MLN model.
    -   `two_axle_ann_mln_model_history.json`: JSON file storing training history for the MLN model.
    -   `two_axle_ann_mln_model.h5`: Trained MLN model for the two-axle vehicle.
    -   `Two_Axle_ANN_MLN.py`: Python script for the Multi-Layer Network (MLN) ANN for the two-axle vehicle.
    -   `two_axle_ann_model_epoch.json`: JSON file storing epoch data for the standard ANN model.
    -   `two_axle_ann_model_history.json`: JSON file storing training history for the standard ANN model.
    -   `two_axle_ann_model.h5`: Trained Keras/TensorFlow standard ANN model for the two-axle vehicle.
    -   `Two_Axle_ANN.py`: Python script for the standard Artificial Neural Network related to the two-axle vehicle.
    -   `Two_Axle_main.m`: Main MATLAB script for the two-axle vehicle simulation.
    -   `Two_Axle_Sim_Data.xlsx`: Excel file containing simulation data for the two-axle vehicle.
    -   `Two_Axle_Simulation_ISO8608.m`: MATLAB script for simulation according to ISO8608 standard for the two-axle vehicle.

## Citation

If you use this work, please cite the following paper:

```bibtex
@article{lin2024deep,
  title={Deep Learning-Based Prediction of Suspension Dynamics Performance in Multi-Axle Vehicles},
  author={Lin, Kai Chun and Lin, Bo-Yi},
  journal={arXiv preprint arXiv:2410.02566},
  year={2024},
  eprint={2410.02566},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

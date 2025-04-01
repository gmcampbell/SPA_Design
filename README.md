**README**: SPA Design

**Accompanying Paper Title**:
Active Learning Design: Modeling Force Output for Axisymmetric Soft Pneumatic Actuators

**Authors**: Gregory M. Campbell, Gentian Muhaxheri, Leonardo Ferreira Guilhoto, Christian D. Santangelo, Paris Perdikaris, James Pikul, and Mark Yim

**Description of the file structure**:

Root file includes .pkl files of characterization (model_data_dictionary) and test dataset (mass_lift_data_dictionary). It also includes general calculations and figures used in the paper.

Data Acquisition includes files relevant to performing characterization and mass testing, including serial communication with relevant firmware.

Hyperparameter Ablations includes results of Weights and Biases ablation study.

Membrane Simulation includes theory-based simulations of membrane expansion and analysis of the results.

Files and variables:

**root**:

model_data_dictionary_N_Pa.pkl: 'pickle' file containing a dictionary of membrane characterization data for 70mm radius Ecoflex 00-30 membranes. Each 'key' contains design parameter data in the form: thickness [mm], contact radius [mm], ring data [mm] (radius_1, width_1, radius_2, width_2). Each 'value' contains a tuple of arrays of membrane test data in the form ((us, ys), Fs, ws) where 'us' [n,7] represent contact height [mm], thickness [mm], contact radius [mm], ring data[mm], 'ys' [n,1] represent pressure [Pa], 'Fs' [n,1] represent contact force [N], and 'ws' represent weights to evenly weight each trial as opposed to each datapoint.

mass_lift_data_dictionary_N_Pa.pkl: 'pickle' file containing a dictionary of membrane test data for 70mm radius Ecoflex 00-30 membranes. Each 'key' contains design parameter data in the form: thickness [mm], contact radius [mm], ring data [mm] (radius_1, width_1, radius_2, width_2). Each 'value' contains a tuple of arrays of membrane test data in the form ((us, ys), Fs) where 'us' [n,7] represent contact height [mm], thickness [mm], contact radius [mm], ring data[mm], 'ys' [n,1] represent pressure [Pa], 'Fs' [n,1] represent force from mass being lifted [N].

acquisition.py - acquisition functions for NN model

archs.py - NN architectures using Flax library

design_optimization.py - helper functions & optimization for NN model

figure_plotting.ipynb - solving for membrane test results reported in paper - creation of paper-relevant and video-relevant figures

membrane_optimization.ipynb - solving for height-maximization membranes

model-baseline_compare - using scikit-learn for k-fold cross-validation to baseline our NN model results. Solving ringless-only NN model error with k-fold cross-validation.

select_next_experiment.ipynb - using model uncertainty to inform choice of next experiments (active learning).

train-utils.py - NN model class and data-loader

**Data Acquisition**: 

Data_Manage.py - helper functions for interacting with .csv files

Run_Experiment.py - executable for running an automated set of data-characterization

Run_Mass_Experiment.py - executable for running a set of membrane mass testing

Serial_Comms.py - helper functions for interfacing with microcontroller via serial cummunication

Test_Procedure.py - helpfer functions for experiments

**Hyperparameter Ablations**: 

ablation_results.ipynb - explanation of hyperparameters with viwewing of dataframe and best parameters

wandb_export_monotonic_actuator_hyperparameter_ablations.csv - data from hyperparameter sweep from Weights and Biases

**Membrane Simulation**:

membrane_simulation.py - functions for executing theoretical simulation.

sim_exp_compare.py - viewing simulation results and error relative to characterization data.

sim_membranes.pkl - 'pickle' file containing a list of simulated test results. Each list item represents a simulation result and is of the form [name, [pressure [pa], force [N], height [m]] where 'name' [6,1] is of the form [thickness [mm], contact radius [mm], ring data [mm] (radius_1, width_1, radius_2, width_2)]. 

sim_results.ipynb - completing simulations for the chaaracterized membranes. Note not all 22 membranes were successfully simulated. Note this script is likely to throw errors, as even the authors only found success of certain PC's. The data from these simulations is found in sim_membranes.pkl.


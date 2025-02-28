"""
Created on April 29, 2024

Define functions for adding data to a CSV file.
Define functions for adding data to .pkl files.
Define functions for reading and plotting data from .pkl files.
"""

import csv
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import math
from bisect import bisect_left

import jax
import jax.numpy as jnp

import imageio

from jax import random
from utils import MLP, RPNEnsemble, RingEncoding, ActuatorModel, Operator
from flax import linen as nn


# Constants
lbf_to_N = 4.44822
PSI_to_Pa = 6894.76
g = 9.81 #m/s^2
in_to_m = 0.0254
mm_to_in = 0.0393701

"""
Perform .unique() on a list of lists.
param seq: [list] The list of lists to be processed
return: [list] A list of unique lists
https://www.peterbe.com/plog/uniqifiers-benchmark - f5
"""
def list_unique(seq, idfun=None): 
   # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    try:
        for item in seq:
            marker = idfun(item)
            if marker in seen: continue
            seen[marker] = 1
            result.append(item)
    except: #deal with lists - seems faulty 8/8/24
        for item in seq:
            marker = idfun(item)
            if tuple(marker) in seen: continue
            seen[tuple(marker)] = 1
            result.append(item)
    return result

"""
Add data line to .csv file.
param filename: [string] The name of the .csv file (including .csv)
param data: [string] The data line to be added to the .csv file
"""
def append_to_csv(filename, data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data])

"""
Format .csv file (remove double quotes, empty rows, and header row).
param filename: [string] The name of the .csv file (including .csv)
"""
def clean_csv(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove double quotes from the line
            line = line.replace('"', '')
            # Check if the line is empty (we accept 'ovf' deal with it in post-processing)
            if line.strip() != '':
                lines.append(line)

    with open(filename, 'w') as file:
        file.writelines(lines)

    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    for i in range(1, len(rows)):
        rows[i][0] = int(rows[i][0]) - int(rows[0][0])
    
    rows[0][0] = 0

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

"""
Sort each dataframe by the sort_value and return a list of dataframes, one for each sort value
param dfs: [list] List of dataframes (can be a list of one dataframe i.e. dfs = list(); dfs.append(df))
param sort_value: [string] The value to sort by
return: [list] List of dataframes
"""
def subset_df(dfs, sort_value):
    df_list = list()
    for i in range(len(dfs)):
        try:
            subsets = len(dfs.iloc[i][sort_value].unique())
            for j in range(subsets):
                df_list.append(dfs[i][dfs[i][sort_value] == dfs[i][sort_value].unique()[j]])
        except:
            # need to deal with lists. probably going to use np.asarray with == and .all()
            unique_data = list_unique(dfs.iloc[i][sort_value])
            print(unique_data)
        
    return df_list

"""
Transfer data from a folder of .csv files to a df (each file becomes a row)
param Directory_Base: [string] The base directory containing the folder
parm folder: [string] The folder containing the .csv files
param Trim_Data: [bool] True to remove rows with near-zero force and values after max force is reached
param Max_Height: [int] A value above the maximum height of the data (i.e. 100 > 70)
param Total_Tests: [int] A value above the total number of tests (i.e. 7 > 3)
"""
def folder_to_df(Directory_path, folder, Trim_Data = False, Max_Height = 100, Total_Tests = 7, Lift_Experiment = False, Leg_Experiment = False, Added_Pressure = False):
    #Make sure we're in the right directory
    init_dir = os.getcwd()
    Directory = os.path.join(Directory_path, folder)
    os.chdir(Directory)

    #define columns
    if Lift_Experiment:
        input_cols = ['time','pressure','height','contact']
        file_cols = ['mass','trial']
    elif Leg_Experiment:
        input_cols = ['time','F_heel','F_Gastroc','pressure','contact']
        file_cols = []
        if Added_Pressure:
            input_cols = ['time','F_heel','F_Gastroc','pressure', 'p1', 'p2','contact']
    else:
        input_cols = ['time','force','pressure','left_height','right_height','flow','contact']
        file_cols = ['nominal_height','trial']
    if Leg_Experiment:
        folder_cols = ['material1','nominal_thickness1','radius1','contact_radius1','A_B1', 'material2','nominal_thickness2','radius2','contact_radius2','A_B2', 'year','month','day']
    else:
        folder_cols = ['material','nominal_thickness','radius','contact_radius','year','month','day','A_B']

    folder_data = folder.split('_')

    data_out = pd.DataFrame(columns=folder_cols + file_cols + input_cols)


    #find every .csv file in folder
    csv_files = [f for f in os.listdir(Directory) if f.endswith('.csv')]
    
    #read in data from .csv files
    for file in csv_files:
        file_data = file.split('_')
        data = pd.read_csv(file, names=input_cols)
        # remove any rows with NaN
        data = data.dropna()
        # remove any rows with ovf pressure
        data = data[data['pressure'] != ' ovf']
        data['pressure'] = data['pressure'].astype(float)
        # remove any rows with near-zero force (out of contact)
        if Trim_Data and not Lift_Experiment:
            data = data[np.abs(data['force']) > 0.05]
        # reformat into a single row
        input_data = pd.DataFrame([data[x].tolist()] for x in input_cols).T
        input_data.columns = input_cols

        BLOWOUT = False
        if not Lift_Experiment and not Leg_Experiment:
            if len(input_data['force'].iloc[0]) > 0:
                F_max = np.min(input_data['force'].iloc[0])
                for i in range(len(input_data['force'].iloc[0])):
                    if i < 4 or i > len(input_data['force'].iloc[0]) - 4:
                            continue
                    # blowout detection (is there a 30% F_max drop?)
                    m1 = np.mean(input_data['force'].iloc[0][i-3:i])
                    m2 = np.mean(input_data['force'].iloc[0][i+1:i+4])
                    if m1 - m2 < 0.7*F_max: #note force values are negative
                        BLOWOUT = True
                    # is this is the last file and not named  '70_2'
                    if file != '70_2.csv':
                        if file == csv_files[-1]:
                            BLOWOUT = True
        if len(folder_data) == 8 or len(folder_data) == 9 or len(folder_data) == 11 or len(folder_data) == 13:
            #add a column for every row of data for the relevant 8 values in folder_data
            for i in range(8):
                #remove the character 't' from the 2nd value in folder_data
                if i in {1, 2}:
                    input_data[folder_cols[i]] = folder_data[i][:-1]
                elif i == 3:
                    input_data[folder_cols[i]] = folder_data[i][:-3]
                else:
                    input_data[folder_cols[i]] = folder_data[i]
        else:
            input_data[folder_cols] = 'unknown'
        if Leg_Experiment:
            input_data['rings1'] = [[folder_data[5], folder_data[6], folder_data[7], folder_data[8]]]
            input_data['rings2'] = [[folder_data[14], folder_data[15], folder_data[16], folder_data[17]]]
        else:
            if len(folder_data) == 13:
                input_data['rings'] = [[folder_data[9], folder_data[10], folder_data[11], folder_data[12]]]
            elif len(folder_data) == 11:
                input_data['rings'] = [[folder_data[9], folder_data[10], np.nan, np.nan]]
            else:  
                input_data['rings'] = [[np.nan, np.nan, np.nan, np.nan]]  # r1,w1,r2,w2

        # NOTE - the 'thicknesses' are not being added to the dataframe during testing, as they require destructive testing.
        # TODO - add a seperate, light, function for adding 'thicknesses' to the dataframe after testing.

        if Lift_Experiment:
            for i in range(len(file_data)):
                if i == 0:
                    input_data[file_cols[i]] = file_data[i]
                else:
                    input_data[file_cols[i]] = file_data[i].split('.')[0]

        elif Leg_Experiment:
            # do nothing
            pass
        
        else:
            for i in range(len(file_data)):
                if i == 1:
                    input_data[file_cols[i]] = file_data[i][0]
                else:
                    input_data[file_cols[i]] = file_data[i].split('.')[0]

        data_out = data_out._append(input_data)
    
    os.chdir(init_dir)
    return data_out

"""
Plot the pressure vs force data for each trial (height)
param df: [dataframe] The dataframe from a single membrane
param title: [string] The title of the plot (usually membrane filename)
param Plot_Final_Only: [bool] True to plot only the final trial for each height
return: [figure, axis] The plot figure and axis. Use plt.show() to display the plot.
"""
def plot_Pressure_Force(df, title ,Plot_Final_Only = True):
    fig, ax = plt.subplots(figsize=(11,7))
    data_out = df

    for i in range(len(data_out)-1):
        #if not Plot_Final_Only or data_out['trial'].iloc[i] == max(data_out['trial'].unique()):
        if not Plot_Final_Only or data_out['trial'].iloc[i] > data_out['trial'].iloc[i+1]:
            pressures = data_out['pressure'].iloc[i]
            if not pressures == []:
                xIntercept = data_out['pressure'].iloc[i][4]
                pressures = [x - xIntercept for x in pressures]
            plt.plot(pressures, data_out['force'].iloc[i], '.', label=data_out['nominal_height'].iloc[i]+' mm')
    
    pressures = data_out['pressure'].iloc[len(data_out)-1]
    if not pressures == []:
        xIntercept = min(x for x in pressures if x > 0.005)
        pressures = [x - xIntercept for x in pressures]
    plt.plot(pressures, data_out['force'].iloc[len(data_out)-1], '.', label=data_out['nominal_height'].iloc[i]+' mm')
    
    maxPressure = max(max(data_out['pressure']))
    plt.plot([0,maxPressure],[0,-maxPressure*float(data_out['contact_radius'].iloc[0])*0.0393701*math.pi**2],'k',label='Force = Pressure*Area')

    fig.gca().invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel('Pressure (PSI)')
    ax.set_ylabel('Force (lbf)')
    plt.legend()
    return fig, ax


"""
Plot the pressure vs force data for each trial (height) - includes PSI to kPa and lbf to N conversions
param df: [dataframe] The dataframe from a single membrane
param title: [string] The title of the plot (usually membrane filename)
param Plot_Final_Only: [bool] True to plot only the final trial for each height
return: [figure, axis] The plot figure and axis. Use plt.show() to display the plot.
"""
def figure_Pressure_Force(df, title, colors ,Plot_Final_Only = True, subsample = 1, new_fig = True):
    if new_fig:
        fig, ax = plt.subplots(figsize=(11,7))
    data_out = df

    PSI_to_kPa = 6.89476
    lbf_to_N = 4.44822

    data_out['pressure'] = data_out['pressure'].apply(np.array)*PSI_to_kPa
    data_out['force'] = data_out['force'].apply(np.array)*-lbf_to_N
    
    for i in range(len(data_out)):

        if not Plot_Final_Only or data_out['trial'].iloc[i] == max(data_out['trial'].unique()):
            index = int(int(data_out['nominal_height'].iloc[i]) / 10)
            if data_out['trial'].iloc[i] == max(data_out[data_out['nominal_height']==data_out['nominal_height'].iloc[i]]['trial'].unique()):
              plt.plot(data_out['pressure'].iloc[i][::subsample], data_out['force'].iloc[i][::subsample], '.', color = colors[index], label=data_out['nominal_height'].iloc[i]+' mm')
            else:
              plt.plot(data_out['pressure'].iloc[i][::subsample], data_out['force'].iloc[i][::subsample], '.', color = colors[index])

    if new_fig:
        ax.set_title(title)
        ax.set_xlabel('Pressure (kPa)')
        ax.set_ylabel('Force (N)')
        plt.legend()
        return fig, ax 
    else:
        return 
    
"""
Plot the pressure vs height data for each trial (mass)
param df: [dataframe] The dataframe from a single membrane
param title: [string] The title of the plot (usually membrane filename)
param Plot_Final_Only: [bool] True to plot only the final trial for each height
param mass_lift_zero: [int] The reading from ToF sensor when the membrane is at zero height new default = 154
    previous default values: 148 1/15/25, 156 8/8/24
return: [figure, axis] The plot figure and axis. Use plt.show() to display the plot.
"""
def figure_Pressure_Height(df, title, Plot_Final_Only = True, subsample = 1, new_fig = True, mass_lift_zero=154):
    if new_fig:
        fig, ax = plt.subplots(figsize=(11,7))
    mass_lift = df
    g = 9.81
    PSI_to_kPa = 6.89476

    mass_lift['pressure'] = mass_lift['pressure'].apply(np.array)*PSI_to_kPa
    mass_lift['trial'] = mass_lift['trial'].apply(int)

    for mass in mass_lift['mass'].unique():
        this_df = mass_lift[mass_lift['mass']==mass]
        for trial in this_df['trial'].unique():
            this_df2 = this_df[this_df['trial']==trial]
            these_pressures = [float(p) for p in this_df2['pressure'][0]]
            these_heights = [(float(h)-mass_lift_zero) for h in this_df2['height'][0]]
            Actual_data_points = len(these_pressures[::subsample])
            if trial == max(mass_lift['trial'].unique()) or not Plot_Final_Only:
                ax.scatter(these_pressures[::subsample], these_heights[::subsample],  label=mass + ' kg')
    
    ax.set_title(title)
    ax.set_xlabel('Pressure (kPa)')
    ax.set_ylabel('Height (mm)')
    plt.legend()
    return fig, ax

"""
Plot the pressure vs. force data for a leg experiment (heel and gastroc)
param df: [dataframe] The dataframe from a single membrane
param title: [string] The title of the plot (usually membrane filename)
param subsample: [int] The number of data points to skip when plotting
param new_fig: [bool] True to create a new figure
return: [figure, axis] Only if new figure. The plot figure and axis. Use plt.show() to display the plot.

"""
def figure_leg_PF(df, title, subsample = 1, color1 = 'b', color2 = 'r', new_fig = True):
    if new_fig:
        fig, ax = plt.subplots(figsize=(11,7))
    data_out = df

    if len(color1) == 1:
        color1 = color1*len(data_out)
    if len(color2) == 1:
        color2 = color2*len(data_out)

    PSI_to_kPa = 6.89476

    data_out['pressure'] = data_out['pressure'].apply(np.array)*PSI_to_kPa
    data_out['F_Gastroc'] = data_out['F_Gastroc'].apply(np.array)
    data_out['F_heel'] = data_out['F_heel'].apply(np.array)
    
    for i in range(len(data_out)):
        plt.plot(data_out['pressure'].iloc[i][::subsample], data_out['F_Gastroc'].iloc[i][::subsample], '.', color = color1[i], label='Gastroc_' + str(i+1))
        plt.plot(data_out['pressure'].iloc[i][::subsample], data_out['F_heel'].iloc[i][::subsample], '.', color = color2[i], label='Heel_' + str(i+1))

    if new_fig:
        ax.set_title(title)
        ax.set_xlabel('Pressure (kPa)')
        ax.set_ylabel('Force (N)')
        plt.legend()
        return fig, ax 
    else:
        return 

"""
Add an individual membrane test to the comprehensive data file and OVERWRITE the old file
param input_df: [dataframe] The dataframe to be added to the comprehensive data file
param Directory_path: [string] The directory containing the comprehensive data file
param old_df_name: [string] The name of the comprehensive data file (including .pkl)
"""
def add_to_data(input_df, Directory_path, old_df_name):
    os.chdir(Directory_path)
    running_df = pickle.load(open(old_df_name, 'rb'))
    # Reject addition of data if it already exists in the comprehensive file (based on 'force' column)
    #NOTE - 'isin' wouldn't work for this.
    for i in range(len(running_df)):
        for j in range(len(input_df)):
            if running_df.iloc[i]['force'] == input_df.iloc[j]['force'] and running_df.iloc[i]['force'] != []: 
                print("This data already exists in the comprehensive file.")
                return
    running_df = running_df._append(input_df)
    running_df.to_pickle(old_df_name)
    return

"""
Load a model with standardized parameters and data (relevant as of 9/2/24)
return: The ensemble used in model creation
NOTE 9/19/24 - this function may already be deprecated. Hyperparameters are changing quickly.
"""
def load_std_ensemble():
    # define model
    # this must use the same hyperparameters as the architecure checkpointed
    # ring encoding was initially 4, [12,5] - lately we've used 12, [16,5]. Note there may also be a problem with whether or not ONE-HOT is used.
    ring_encoder = RingEncoding(12,
                                MLP([16,5]),
                                na_embeding_init = lambda key, shape, dtype=float : random.uniform(key, shape=shape, dtype=dtype, minval=-jnp.sqrt(3), maxval=jnp.sqrt(3)),
                                )
    arch = ActuatorModel(ring_encoder,
                        MLP([24,16], output_activation=nn.gelu),
                        polynomial_degree=2)
    ensemble_size = 42 # try reducing this value if you run into 'RESOURCE_EXHAUSTED' errors. This should probably not be lower than 32ish
    ensemble = RPNEnsemble(arch, ensemble_size)
    return ensemble


"""

Reformat dataframe to incorporate correct data types in each column
param df: [dataframe] The dataframe to be reformatted
return: [dataframe] The reformatted dataframe

"""
def reformat_df(data):
    # converts entries from string into float and lists into jax arrays
    # possible force values contained:
    force_names = ['F_Gastroc', 'F_heel']
    float_names = ['nominal_height', 'nominal_thickness', 'contact_radius', 'trial']
    for name in float_names:
        if name in data.columns:
            data[name] = data[name].apply(float)
    if 'force' in data.columns:
        data['force'] = data['force'].apply(lambda x : -jnp.array(x)) 
    for force in force_names:
        if force in data.columns:
            data[force] = data[force].apply(lambda x : jnp.array(x))    
    if 'pressure' in data.columns:
        data['pressure'] = data['pressure'].apply(jnp.array)
    if 'rings' in data.columns:
        data['rings'] = data['rings'].apply(lambda x: jnp.array([float(i) for i in x]))

    return data

"""

Transform data from a dataframe into a format that can be used by the model
param data: [dataframe] The dataframe to be transformed
return: [tuple] The transformed data - (us, ys, fs,ws) - ready to be used for model training

"""

def transform_data(data):
    us, ys, forces = [], [], []
    us = jnp.hstack((jnp.array(data.nominal_height.values)[:,None],
                    jnp.array(data.nominal_thickness.values)[:,None],
                    jnp.array(data.contact_radius.values)[:,None],
                    jnp.stack(list(data.rings))))

    ys = list(data.pressure.values)
    fs = list(data.force.values)

    organized_us = []
    organized_ws = []

    for j in range(len(us)):
        # number of pressure measurements should be equal to force measurements
        assert len(ys[j]) == len(fs[j])
        num_measurements = len(ys[j])
        organized_us.append(jnp.tile(us[j], (num_measurements,1)))
        organized_ws.append(jnp.ones((num_measurements,1))*1e3/num_measurements)
    return ((jnp.concatenate(organized_us),jnp.concatenate(ys)[:,None]),
                          jnp.concatenate(fs)[:,None],
                          jnp.concatenate(organized_ws))



"""
# https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value

Assumes myList is sorted. Returns closest value to myNumber.

If two numbers are equally close, return the smallest number.
"""
def take_closest(myList, myNumber):

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], pos
    if pos == len(myList):
        return myList[-1], pos - 1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after, pos
    else:
        return before, pos - 1
    

""""""

"""
Transfer membrane representation from model form to simulation form
Input: membranes as [thickness, outer radius, inner radius, [rings]] (only works for 0 or 2 rings !)
Output: membranes as [height, change_material, material_type]
"""
def mem_to_input(mem):
    h = mem[0]
    if jnp.isnan(jnp.array(mem[3])).any(): # 0 rings (ignoring 1 ring)
        change_material = (mem[2], mem[1])
        material_type = ('elast',)
    else: # 2 rings
        change_material = (mem[2], mem[3][0]-mem[3][1], mem[3][0] + mem[3][1], mem[3][2] - mem[3][3], mem[3][2]+mem[3][3], mem[1])
        material_type = ('elast', 'stiff', 'elast', 'stiff', 'elast')
    
    return h, change_material, material_type



# Leo's ring generation: ~~~~~~~~~~~~~~~~~~~~~~~
# computes radii and widths of rings

# Parameters for material
_radius = 70. # radius of the entire membrane
_min_spacing = 3. # minimum spacing between rings and other rings/boundaries
_min_width = 5. # minimum width of each ring

_tmin = 1. # minimum membrane thickness
_tmax = 3. # maximum membrane thickness
_contactmin = 25.4 # minimum contact disk radius
_contactmax = 38.1  # maximum contact disk radius

# effectively, this function transforms numbers between 0 and 1 into the proper ring parameters
def get_rw(w0, ring_params, num_rings, radius=_radius, min_spacing=_min_spacing, min_width=_min_width, max_rings=2):
    ring_params = ring_params.sort() # sorts entries from smallest to largest
    available_space = radius - w0 - min_spacing*(num_rings+1) - 2*min_width*num_rings
    #assert (available_space>0), "There is no space available for rings!"
    radii = [w0 + (i+1)*min_spacing + (2*i+1)*min_width +  available_space*(ring_params[2*i+1]+ring_params[2*i])/2 for i in range(num_rings)]
    widths = [min_width + available_space*(ring_params[2*i+1]-ring_params[2*i])/2 for i in range(num_rings)]
    radii = jnp.array(radii + (max_rings-num_rings)*[jnp.nan])
    widths = jnp.array(widths + (max_rings-num_rings)*[jnp.nan])
    return jnp.ravel(jnp.vstack([radii, widths]), order='F')
    
# 'optimization variables to physical variables'
def opt_var_to_phys_var(membrane_coefs, num_rings, max_rings=2):
    # membrane_coefs is a vector with up to 6 entries: t, w0, r1, w1, r2, w2
    material_params = membrane_coefs[:2]
    ring_params = jnp.concatenate([membrane_coefs[2:], jnp.array(2*(max_rings-num_rings)*[jnp.nan])])
    ring_params = get_rw(material_params[1], ring_params, num_rings)

    membrane = jnp.concatenate([material_params, ring_params])
    return membrane

# Attempt to do the inverse of the 'get_rw' function
def reverse_get_rw(rw, w0, num_rings, radius=_radius, min_spacing=_min_spacing, min_width=_min_width, max_rings=2):
    # Reshape the input array to separate radii and widths
    rw = jnp.reshape(rw, (2, max_rings), order='F')
    radii = rw[0, :num_rings]
    widths = rw[1, :num_rings]
    
    # Calculate available space
    available_space = radius - w0 - min_spacing * (num_rings + 1) - 2 * min_width * num_rings
    
    # Reverse the calculation of ring_params
    ring_params = []
    for i in range(num_rings):
        ring_param1 = (2 * (radii[i] - w0 - (i + 1) * min_spacing - (2 * i + 1) * min_width) / available_space) - widths[i]
        ring_param2 = (2 * widths[i] / available_space) + ring_param1
        ring_params.extend([ring_param1, ring_param2])
    
    return jnp.array(ring_params)

'''
Takes: dataframe with 'force' , 'pressure' , 'nominal_height' columns (lbf, psi, mm)
Returns: forces, pressures, heights as flat arrays (N, Pa, mm)
'''
def df_to_flat(new_df):
    # make plottable p, h, F
    # make pressure, force, height values from new_df and concatenate them into a single array
    pressures = [np.array(new_df['pressure'].iloc[i]) for i in range(len(new_df))]
    forces = [np.array(new_df['force'].iloc[i]) for i in range(len(new_df))]
    heights = [np.array(np.tile(new_df['nominal_height'].iloc[i],len(new_df['force'].iloc[i]))) for i in range(len(new_df))]
    # concatenate the lists
    pressures = np.concatenate(pressures)*PSI_to_Pa
    forces = np.concatenate(forces)*lbf_to_N
    heights = np.concatenate(heights)

    return pressures, forces, heights

'''
Takes: completed plot (axs), and a boolean to save the plot as a gif
Returns: a list of frames if save is True (can be used to create a gif with imageio.mimsave)
'''
def plot_spin(axs, save = False):
    frames = []
    for angle in range(0, 360 + 1):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180

        # Cycle through a full rotation of azimuth
        elev = azim = roll = 0
        azim = angle_norm

        # Update the axis view and title
        for ax in axs:
            ax.view_init(elev, azim, roll)
        # plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

        if save:
            plt.savefig(f'frame_temp.png')
            frames.append(imageio.v3.imread(f'frame_temp.png'))
        else:
            plt.draw()
            plt.pause(.001)
    if save:
        return frames
"""
Created on July 31, 2024
Gregory Campbell

This file is the place to input test- or computer-specific parameters and run the mass lift test procedure.

The output is a set of .csv files, a plot, and a .pkl file containing the data.
"""

import os
import cv2
import threading
import matplotlib.pyplot as plt
from Test_Procedure import run_mass_experiment
from Data_Manage import folder_to_df, figure_Pressure_Height

"""
Test-specific variables set here
"""
# Title
Material = 'E0030'
Thickness = '2t'
Radius = '70r'
Contact = '25.4cap'
Date = '25_01_30' # yr_m_day
Sample = 'A' # A/B depends on casting location
Trial = '1.0'
Rings = '33.4_5_46.4_5' # Radius_Width_Radius_Width

# Concatenate
Sample_String = Material + '_' + Thickness + '_' + Radius + '_' + Contact + '_' + Date + '_' + Sample + '_' + Trial + '_' + Rings

print(Sample_String)

# Tests per mass
Num_Tests = 2
Masses = [1.5, 2.5, 4] #Note: 1kg is approx. mass of 1-DoF teststand hardware (8/8/24)

# Record values in .csv for duration of deflation (false = just inflation)
MEASURE_DEFLATION = True
"""
Global / computer-specific variables set here

We found it helpful to use a User variable when alternating between computers.
"""

# User = 'Jason' # 'Greg' or 'Jason'
User = 'Greg'
LAPTOP_CAM = False # True if using laptop webcam, False if using external webcam
# NOTE - ALSO CHANGE YOUR COM PORT IN Test_Procedure.py when changing user!!

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This section sets output path (and) - update for individual PC.
if User == 'Jason':
    parent_dir = 'C:\\Users\\jason\\Desktop\\ModLab\\Serial_Experiment_Control\\data'
if User == 'Greg':
    parent_dir = 'C:\\Users\\campb\\Python_Code\\data'
os.chdir(parent_dir)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This section defines global variables
if User == 'Jason':
    serial_port = 'COM3'
    baud_rate = 115200
    Webcam_Value = 1 #only works if internal webcam activated (0 in VSCode)
if User == 'Greg':
    serial_port = 'COM3'
    baud_rate = 115200
    Webcam_Value = 0 #only works if internal webcam activated (0 in VSCode)

RECORD_BOOL = False # This is a 'state machine' for when to record video

"""
Function for video recording
param cap: [cv2.VideoCapture] The video capture object
param out: [cv2.VideoWriter] The video writer object
"""
# This section records video:
def record_video(cap,out):
    
    try:
        while (RECORD_BOOL):
            ret, frame = cap.read()
            # replace frame with 'rotated' depending on camera orientation.
            rotated=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            if ret:
                # out.write(rotated)
                out.write(frame)
    except KeyboardInterrupt:
        cap.release()
        out.release()
        print('Video stopped by user')


"""
Main Code
"""

data_path = os.path.join(parent_dir, Sample_String)
vid_path = os.path.join(data_path,'videos')
try:
    os.mkdir(data_path)
except:
    print("Data folder has been created already")
    input("Press Enter to continue...")
try:
    os.mkdir(vid_path)
except:
    print("Videos folder already created")

# Webcam Startup - create a VideoCapture object to access the webcam (0 is usually the default camera)
ret = 0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
cap = cv2.VideoCapture(Webcam_Value)
while not ret:
    ret, frame = cap.read()

frame_size = (int(cap.get(3)),int(cap.get(4))) # 3, 4 fits our camera orientation.
if LAPTOP_CAM:
    frame_size = (int(cap.get(4)),int(cap.get(3))) # 4, 3 fits alternative camera orientation.


# Primary Loop:
for Mass in Masses:
    print(f"Mass {Mass} started")
    for Test_Num in range(Num_Tests):
        print(f"Trial {Test_Num+1}")
        input("Press enter to start recording and writing data to csv file...")
        save_to = os.path.join(vid_path,str(Mass)+str(Test_Num)+'.mp4')
        out = cv2.VideoWriter(save_to, fourcc, 30.0, frame_size)  # 30 fps
        
        # Thread 'x': Record
        RECORD_BOOL = True
        x = threading.Thread(target=record_video,args=(cap,out,))
        x.start()
        
        # Thread 'y': Test
        y = threading.Thread(target=run_mass_experiment,args=(Test_Num,data_path,serial_port,baud_rate,MEASURE_DEFLATION,Mass))
        y.start()
        
        # Finish Test & Recording
        y.join()
        RECORD_BOOL = False
        x.join()
        out.release()
        print("Trial finished")
    print(f"Mass {Mass} finished")

# Close out camera
cap.release()

print("Testing Completed.")

os.chdir(data_path)

# Combined all data into a single .pkl file
df = folder_to_df(parent_dir, Sample_String, Trim_Data=True, Lift_Experiment = True)
# Save data to .pkl
df.to_pickle(Sample_String+'_data.pkl')

# Plot data
fig, ax = figure_Pressure_Height(df, Sample_String, Plot_Final_Only=True)
plt.savefig(Sample_String+'_plot.png')
plt.show()
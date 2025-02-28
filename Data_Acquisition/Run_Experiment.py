"""
Created on April 29, 2024
Gregory Campbell - Edits by Jason Matthew

This file is the place to input test- or computer-specific parameters and run the test procedure.

The output is a set of .csv files, a plot, and a .pkl file containing the data.
"""

import os
import cv2
import threading
import matplotlib.pyplot as plt
from Test_Procedure import set_height, run_experiment
from Data_Manage import folder_to_df, plot_Pressure_Force, add_to_data

"""
Test-specific variables set here
"""
# Title
Material = 'E0030'
Thickness = '3.0t'
Radius = '70r'
Contact = '25.4cap'
Date = '25_01_07'# yr_m_day
Sample = 'A' # A/B depends on casting location
Trial = '3.1'
Rings = '33_5_61.5_5'

Sample_String = Material + '_' + Thickness + '_' + Radius + '_' + Contact + '_' + Date + '_' + Sample + '_' + Trial + '_' + Rings

print(Sample_String)

# Tests per height
Num_Tests = 3
# Height list
Test_Heights = [0,10,20,30,40,50,60,70] #mm
New_Heights = []

# Record values in .csv for duration of deflation (false = just inflation)
MEASURE_DEFLATION = True

"""
Global / computer-specific variables set here

We found it helpful to use a User variable when alternating between computers.
"""

User = 'Greg' # 'Greg' or 'Jason'
# NOTE - ALSO CHANGE YOUR COM PORT IN Test_Procedure.py when changing user!!

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This section sets output path (and) - update for individual PC.
if User == 'Jason':
    parent_dir = 'C:\\Users\\jason\\Desktop\\ModLab\\Serial_Experiment_Control\\data'
if User == 'Greg':
    parent_dir = 'C:\\Users\\campb\\Python_Code\\data'
os.chdir(parent_dir)
Complete_Data = '8_28_2024_data_subset.pkl' # If compiling to a complete data set
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This section defines global variables
if User == 'Jason':
    serial_port = 'COM3'
    baud_rate = 115200
    Webcam_Value = 1 #only works if internal webcam activated (0 in VSCode)
if User == 'Greg':
    serial_port = 'COM5'
    baud_rate = 115200
    Webcam_Value = 0 #only works if internal webcam activated (0 in VSCode)

# Median burst pressures from unringed testing - will use a safety factor (*0.7) in initial inflations
if Material == 'E0030' or Material == 'E30':
    Burst_Pressure = 0.89 # PSI
elif Material == 'D10':
    Burst_Pressure = 2.47 # PSI
elif Material == 'D30':
    Burst_Pressure = 5.13 # PSI
else:
    input("Unknown Material. Press Enter to continue...")
    Burst_Pressure = 500 # PSI

# approximate motor rates:
up_rate = 15.0 # mm/s 
down_rate = 16.0 # mm/s

# Local maxima
max_recordable_height = 400 #mm

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
            rotated=cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            if ret:
                out.write(rotated)
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

frame_size = (int(cap.get(4)),int(cap.get(3))) # would normally be 3,4 - this fits our camera orientation.


# Primary Loop:
for height in Test_Heights:
    set_height(height,serial_port,baud_rate,up_rate,down_rate,max_recordable_height)
    print(f"Setting height to {height} mm")
    for Test_Num in range(Num_Tests):
        save_to = os.path.join(vid_path,str(height)+'1_'+str(Test_Num)+'.mp4')
        out = cv2.VideoWriter(save_to, fourcc, 30.0, frame_size)  
        
        # Thread 'x': Record
        RECORD_BOOL = True
        x = threading.Thread(target=record_video,args=(cap,out,))
        x.start()
        
        # Thread 'y': Test
        print(f"Testing at {height} mm, Test {Test_Num}")
        y = threading.Thread(target=run_experiment,args=(height,Test_Num,data_path,serial_port,baud_rate,0.7*Burst_Pressure,MEASURE_DEFLATION,New_Heights))
        y.start()
        
        # Finish Test & Recording
        y.join()
        print(f"{New_Heights}")
        RECORD_BOOL = False
        x.join()
        out.release()
        print("Trial finished")

New_Heights.reverse()
print(f"Testing these heights again with a higher burst pressure: {New_Heights}")

# Secondary Loop:
for height in New_Heights:
    set_height(height,serial_port,baud_rate,up_rate,down_rate,max_recordable_height)
    print(f"Setting height to {height} mm")
    for Test_Num in range(Num_Tests):
        save_to = os.path.join(vid_path,str(height)+'2_'+str(Test_Num)+'.mp4')
        out = cv2.VideoWriter(save_to, fourcc, 30.0, frame_size)  # 30 fps
        
        # Thread 'x': Record
        RECORD_BOOL = True
        x = threading.Thread(target=record_video,args=(cap,out,))
        x.start()
        
        # Thread 'y': Test
        print(f"Testing at {height} mm, Test {Test_Num}")
        y = threading.Thread(target=run_experiment,args=(height,Num_Tests+Test_Num,data_path,serial_port,baud_rate,Burst_Pressure,MEASURE_DEFLATION))
        y.start()
        
        # Finish Test & Recording
        y.join()
        RECORD_BOOL = False
        x.join()
        out.release()
        print("Trial finished")

        
# Close out camera
cap.release()

print("Testing Completed.")

#ADD IMPORTANT CODE HERE

# Combined all data into a single .pkl file
df = folder_to_df(parent_dir, Sample_String, Trim_Data=True)
# Save data to .pkl
df.to_pickle(Sample_String+'_data.pkl')


# Plot data
fig, ax = plot_Pressure_Force(df, Sample_String, Plot_Final_Only=True)
plt.savefig(Sample_String+'_plot.png')
plt.show()

# Add data to the running test dataframe .pkl
os.chdir(parent_dir)
Store_Data = input("Store data in comprehensive file? (y/n): ")  
if Store_Data in {'y','Y'} and os.path.exists(Complete_Data):
    add_to_data(df, parent_dir, Complete_Data)
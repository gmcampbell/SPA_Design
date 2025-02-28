"""
Created on April 28, 2024
Gregory Campbell

Define functions for testing control via serial communications.
"""

from Serial_Comms import *
from Data_Manage import append_to_csv, clean_csv
import time
import os
import serial
import numpy as np

# This section defines global variables - it must be updated when running on a new PC.
serial_port = 'COM5'
baud_rate = 115200

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function definitions

"""
Move linear actuators for a specified amount of time.
param run_time: The amount of time to run the linear actuators (seconds).
param direction: The direction in which to move the linear actuators ('up' or 'down').
param serial_port: The serial port to which the test rig controller is connected.
param baud_rate: The baud rate for serial communications.
"""
def move_for(run_time, direction, serial_port1, baud_rate1=115200):
    serialPort = serial.Serial(
            port=serial_port, baudrate=baud_rate, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
            )
    time.sleep(1)
    start_time = time.time()
    run_time = min(run_time,0.5) # maximum run time of 0.5 second at a time
    
    if direction == 'up':
        # print("Going up")
        serialPort.write(BOTH_PULL)
        #send_character(serial_port, baud_rate, BOTH_PULL)
    if direction == 'down':
        # print("Going down")
        serialPort.write(BOTH_PUSH)
        #send_character(serial_port, baud_rate, BOTH_PUSH)

    try:
        while(time.time()-start_time < run_time):
            # make sure there isn't any force
            # read in serial data

            time.sleep(0.01)
    except KeyboardInterrupt:
        interrupt_stop(serial_port, baud_rate)
        print('Stopped by user')

    #send_character(serial_port, baud_rate, BOTH_STOP)
    # print("Stopping")
    serialPort.write(BOTH_STOP)
    serialPort.close()

"""
Set the height of the test rig to a desired height.
param desired_height_mm: The desired height from the top of the membrane in mm.
param serial_port: The serial port to which the test rig controller is connected.
param baud_rate: The baud rate for serial communications.
param up_rate: The rate at which the test rig moves upward.
param down_rate: The rate at which the test rig moves downward.
"""
def set_height(desired_height_mm, serial_port, baud_rate=115200, up_rate=15, down_rate=16, max_recordable_height=400, prev_error=0):
    # Note desired height is a single value, as it should be equal for both.
    tolerance = 1  # Tolerance level in mm

    if prev_error > tolerance:
        run_time = abs(prev_error)/down_rate
        move_for(run_time, 'up', serial_port, baud_rate)
    if prev_error < -tolerance:
        run_time = abs(prev_error)/up_rate
        move_for(run_time, 'down', serial_port, baud_rate)
    
    # Read current height
    L_Height, R_Height = read_current_height(serial_port, baud_rate)

    # Deal with faulty sensor:
    if R_Height > max_recordable_height:
            ToF_Reset(serial_port, baud_rate)
            set_height(desired_height_mm, serial_port, baud_rate, up_rate, down_rate)
            return
    # Adjust height (simple MPC) based on sensor readings:
    else:
        error_R = desired_height_mm - R_Height

        if abs(error_R) <= tolerance:
            L_Height, R_Height = read_height_for(5,serial_port, baud_rate,max_recordable_height)
            if R_Height > max_recordable_height:
                ToF_Reset(serial_port, baud_rate)
                set_height(desired_height_mm, serial_port, baud_rate, up_rate, down_rate)
            return #completed.

        else:
            set_height(desired_height_mm, serial_port, baud_rate, up_rate, down_rate, max_recordable_height, error_R)
            return

"""
Somewhat arbitrary affine function to return a maximum record time based on what height load cell is set to.
"""
def time_eqn(height, max_record_time=150):
    experiment_time = 30+2*height # Note this is max time (hoping to hit the switch/pressure limit before this time)
    return min(experiment_time,max_record_time)

"""
Main function to run an experiment (inflate and deflate at a specific height, write data to a .csv)
param height_number: The height of the test rig in mm.
param test_number: The number of the current trial (generally 0-2)
param data_path: The path to the directory where data should be saved (os.path.join(parent_dir, sample_string))
param Burst_Pressure: The pressure at which the test rig should stop inflating (PSI), see initial test set for material values
param MEASURE_DEFLATION: Whether or not to measure deflation (True/False)
"""
def run_experiment(height_number,test_number,data_path,serial_port,baud_rate=115200,Burst_Pressure=0.5,MEASURE_DEFLATION=True,Next_Heights=[]):
    serialString = ""  # Used to hold data coming over UART
    
    this_experiment = str(height_number) + "_" + str(test_number)
    this_trial_path = os.path.join(data_path,this_experiment)
    file_name = this_trial_path + ".csv"

    # Try to get height values online
    ESP_Hard_Reset(serial_port, baud_rate)

    # Open serial port (we will hold it open for the duration of the experiment)
    serialPort = serial.Serial(
        port=serial_port, baudrate=baud_rate, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
    )
    time.sleep(1)
    
    # Begin inflation
    serialPort.write(AIR_PUMP_ON)
    # print("Air pump on.")
    time.sleep(0.02)
    start_time = time.time()

    burst_pressure_reached = False
    bPresNum = 0
    # Record data throughout inflation
    try:
        while (time.time()-start_time < time_eqn(height_number)):
            try:
                # Wait until there is data waiting in the serial buffer
                if serialPort.in_waiting > 0:
            
                    # Read data out of the buffer until a carraige return / new line is found
                    serialString = serialPort.readline()
                    
            
                    # Print the contents of the serial data
                    if(not serialString == b'' ):
                        try:
                            append_to_csv(file_name,serialString.decode("Ascii"))
                        except:
                            pass
                    values = parse_string(str(serialString))
                    # Check for contact
                    if(0 == float(values[-1])):
                        break
                    # Check for pressure stop
                    # print("Pressure: " + values[2])
                    if(float(values[2]) >= Burst_Pressure):
                        print("Burst pressure reached.")
                        bPresNum += 1
                        if bPresNum == 5: #High pressure values from pressure sensors don't go past the first 3 reads usually
                            burst_pressure_reached = True
                            break
            except KeyboardInterrupt:
                interrupt_stop(serial_port, baud_rate)
                print('Stopped by user')
    except KeyboardInterrupt:
        serialPort.close()
        interrupt_stop(serial_port, baud_rate)
        print('Stopped by user')
    
    # Stop inflation and release air
    serialPort.write(AIR_PUMP_OFF)
    # print("Air pump off.")
    time.sleep(0.25)
    serialPort.write(AIR_RELEASE_ON)
    
    # Deflate time equal to inflate time
    start_time = time.time()
    double_tap = False
    Stop_Record = False
    stopDeflate = 0
    manualOverride = 0
    try:
        while (time.time()-start_time < time_eqn(height_number)):
            try:
                if serialPort.in_waiting > 0:
                    line = serialPort.readline()
                    values = parse_string(line)

                    if MEASURE_DEFLATION:
                        if(not line == b'' and not Stop_Record): #TODO check if stop_record is necessary/works
                            try:
                                append_to_csv(file_name,line.decode("Ascii"))
                            except:
                                pass
                    
                    if len(values) >= 6:
                        P_PSI = float(values[2])
                        if(P_PSI <= 0.001):
                            if stopDeflate >= 5: #avoid a noise-based reading. -GMC 8/29/24
                                break
                            stopDeflate += 1
                        #Stop recording on contact break:
                        F_up = float(values[1])
                        if F_up>=-0.04 and not Stop_Record: 
                            Stop_Record = True

                        # Manual Override for Deflation
                        if (np.logical_and(float(values[6]) == 0, P_PSI < 0.3)):
                            if manualOverride == 15: #Doesnt usually press the limit switch longer than 15 reads
                                break
                            manualOverride += 1
                    if time.time()-start_time >= 1 and values[0]==0 and not double_tap:
                        serialPort.write(AIR_RELEASE_ON)
                        double_tap = True
            except Exception as e:
                print("Error reading data:", e)
            except KeyboardInterrupt:
                interrupt_stop(serial_port, baud_rate)
                print('Stopped by user')
    except KeyboardInterrupt:
        interrupt_stop(serial_port, baud_rate)
        print('Stopped by user')

    #send_character(serial_port, baud_rate, AIR_RELEASE_OFF)   
    time.sleep(1)
    serialPort.write(AIR_RELEASE_OFF)
    serialPort.close()
    
    if burst_pressure_reached and (height_number not in Next_Heights):
        Next_Heights.append(height_number)
        
    # Note - the following only works if you strap pin 15 of ESP32 to GND (supressing init message).
    clean_csv(file_name)

def run_mass_experiment(test_number,data_path,serial_port,baud_rate=115200,MEASURE_DEFLATION=True,mass=''):
    serialString = ""  # Used to hold data coming over UART
    
    this_experiment = str(mass) + "_" + str(test_number)
    this_trial_path = os.path.join(data_path,this_experiment)
    file_name = this_trial_path + ".csv"

    # Try to get height values online
    ESP_Hard_Reset(serial_port, baud_rate)

    # Open serial port (we will hold it open for the duration of the experiment)
    serialPort = serial.Serial(
        port=serial_port, baudrate=baud_rate, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
    )
    time.sleep(1)
    
    # Begin inflation
    serialPort.write(AIR_PUMP_ON)
    time.sleep(0.02)
    start_time = time.time()

    # Record data throughout inflation
    try:
        while (time.time()-start_time < 150): #150 = max record time
            try:
                # Wait until there is data waiting in the serial buffer
                if serialPort.in_waiting > 0:
            
                    # Read data out of the buffer until a carraige return / new line is found
                    serialString = serialPort.readline()
                    
                    # Print the contents of the serial data 
                    if(not serialString == b'' ):
                        try:
                            append_to_csv(file_name,serialString.decode("Ascii"))
                        except:
                            pass
                    values = parse_string(str(serialString))
                    values[-1] = values[-1][0]
                    
                    # Check for contact (manual input)
                    if(0 == float(values[-1])):
                        print("Limit switch has been pressed, deflate membrane.")
                        break
            except KeyboardInterrupt:
                interrupt_stop(serial_port, baud_rate)
                print('Stopped by user')
    except KeyboardInterrupt:
        serialPort.close()
        interrupt_stop(serial_port, baud_rate)
        print('Stopped by user')
    
    # Stop inflation and release air
    # No active air release.
    serialPort.write(AIR_PUMP_OFF)

    time.sleep(2)

    # Deflate time equal to inflate time
    start_time = time.time()
    Stop_Record = False
    prevVal = 1000
    try:
        while (time.time()-start_time < 150): #150 = max record time
            try:
                if serialPort.in_waiting > 0:
                    line = serialPort.readline()
                    values = parse_string(str(line))
                    values[-1] = values[-1][0]

                    if MEASURE_DEFLATION:
                        if(not line == b'' and not Stop_Record):
                            try:
                                append_to_csv(file_name,line.decode("Ascii"))
                            except:
                                pass
                    
                    if (int(values[-1]) > prevVal):
                        print("Limit switch has been pressed, stopping trial.")
                        break
                    else:
                        prevVal = int(values[-1])
            except Exception as e:
                print("Error reading data:", e)
            except KeyboardInterrupt:
                interrupt_stop(serial_port, baud_rate)
                print('Stopped by user')
    except KeyboardInterrupt:
        interrupt_stop(serial_port, baud_rate)
        print('Stopped by user')

    time.sleep(1)
    serialPort.close()
    
    # Note - the following only works if you strap pin 15 of ESP32 to GND (supressing init message).
    clean_csv(file_name)


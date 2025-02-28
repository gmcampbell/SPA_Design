"""
Created on April 28, 2024
Gregory Campbell

Define variables for custom communication protocol.
Define functions for serial communications.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# imports
import serial
import time
import statistics
import pandas as pd

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Defines commands known by the test setup
AIR_PUMP_ON = 'w\n'.encode()
AIR_PUMP_OFF = 's\n'.encode()
AIR_RELEASE_ON = 'q\n'.encode()
AIR_RELEASE_OFF = 'a\n'.encode()
L_LIN_PUSH = 'u\n'.encode()
L_LIN_PULL = 'i\n'.encode()
L_LIN_STOP = 'o\n'.encode()
R_LIN_PUSH = 'j\n'.encode()
R_LIN_PULL = 'k\n'.encode()
R_LIN_STOP = 'l\n'.encode()
RESET_ESP = 'm\n'.encode()
BOTH_PUSH = 'y\n'.encode()
BOTH_STOP = 'h\n'.encode()
BOTH_PULL = 'n\n'.encode()
SENSOR_POWER_ON = '9\n'.encode()
SENSOR_POWER_OFF= '0\n'.encode()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function definitions

"""
Perform an affine transformation on the ToF sensor reading to convert it to a height in mm.
The affine transformation is determined by the calibration data, which is determined by the 'Test_Rig_Calibration.py' script.
param Tof_Reading: The ToF sensor reading to be transformed.
param LR: The side of the test rig the reading is from. 'L' for left, 'R' for right.
return: [float] The height in mm corresponding to the ToF sensor reading.
"""
def affine_transform(ToF_Reading,LR,max_recordable_height=400):

    # Input from 'Test_Rig_Calibration.py' - updated 11/26/24
    L_coefficients = 12.715710953923296 + 0.6092962579084987*ToF_Reading
    R_coefficients = -87.35454856094495 + 0.998299015527841*ToF_Reading
    pressure_chamber_zero = 42.0

    if ToF_Reading >= max_recordable_height:
        return 999
    if LR == 'L':
        # return 1.07*ToF_Reading-78.0 - pressure_chamber_zero #11/16/23 Calibration
        # return 1.05*ToF_Reading-78.5 - pressure_chamber_zero #01/22/24 Calibration
        return L_coefficients - pressure_chamber_zero
    if LR == 'R':
        # return 1.0*ToF_Reading-80.1 - pressure_chamber_zero #11/16/23 Calibration
        # return 1.0*ToF_Reading-82.6 - pressure_chamber_zero #01/22/24 Calibration
        return R_coefficients - pressure_chamber_zero
    else:
        print("Error in affine transform: specify L or R")
        return 999

"""
Send a command (character) to the test setup via serial communications.
param serial_port: The serial port to which the test rig controller is connected.
param baud_rate: The baud rate of the serial connection.
param character: The command to be sent to the test setup.
"""
def send_character(serial_port, baud_rate=115200, character='\n'.encode()):
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        
        # Write the specified character to the serial line
        ser.write(character)
        # print(f"Sent {character}")
        
        # Close the serial port
        ser.close()
        
        time.sleep(0.02)
        
    except serial.SerialException as e:
        print(f"An error occurred: {e}")

"""
Take in a string and return a list of strings
param string: The string to be parsed.
return: A list of strings.
Note: unclear if the 'str_line[2:-3]' is necessary, but it is used in the original code. Altered for leg lifts 9/17/24.
Re Note: the -3 is to deal with '0\\n' - easier to just use values[-1] = values[-1][0]
"""
def parse_string(string, leg_lift=False):
    str_line = str(string)
    if leg_lift:
        pass
    else:
        str_line = str_line[2:-3]
    values = list(map(str.strip, str_line.split(',')))
    return values

"""
Red a string from the serial port and parse it into a list of values.
param serial_port: The serial port to which the test rig controller is connected.
param baud_rate: The baud rate of the serial connection.
return: A list of [string] values read from the serial port.
"""
def read_state(serial_port, baud_rate=115200):
    state = pd.DataFrame(columns=['Time','Force','Pressure','L_Height','R_Height','Flow','Limit_Switch'])
    serialPort = serial.Serial(
        port=serial_port, baudrate=baud_rate, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
    )
    time.sleep(0.1)
    if serialPort.in_waiting > 0:
        # Read data out of the buffer until a carraige return / new line is found
        serialString = serialPort.readline()
        # Print the contents of the serial data
        if(not serialString == b''):
            try:
                str_line = str(serialString)
                values = parse_string(str_line)
                if len(values) > 6:
                    serialPort.close()
                    return values
                else:
                    #try 1 more time
                    serialString = serialPort.readline()
                    str_line = str(serialString)
                    values = parse_string(str_line)
                    serialPort.close()
                    return values
            except:
                return 0

"""
Read the height from the serial port for a specified amount of time.
param read_time: The amount of time to read the height from the serial port (seconds)
param serial_port: The serial port to which the test rig controller is connected.
param baud_rate: The baud rate of the serial connection.
param max_recordable_height: The maximum height that can be recorded by the ToF sensor (must be less than error val of 999).
return: [float] The average height of the left and right ToF sensors over the specified time period (transformed via affine transform).
"""
def read_height_for(read_time, serial_port, baud_rate=115200, max_recordable_height=400):
    # print('Reading height for ' + str(read_time) + ' seconds')
    start_time = time.time()
    L_vals = []
    R_vals = []
    serialPort = serial.Serial(
        port=serial_port, baudrate=baud_rate, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
    )
    time.sleep(0.1)

    # read the serial port for x seconds
    try:
        while (time.time()-start_time < read_time):
            # Wait until there is data waiting in the serial buffer
            if serialPort.in_waiting > 0:
                # Read data out of the buffer until a carraige return / new line is found
                serialString = serialPort.readline()
                # Print the contents of the serial data
                if(not serialString == b''):
                    try:
                        str_line = str(serialString)
                        values = parse_string(str_line)
                        if len(values) >= 6:
                            L_Height = float(values[3])  # Height in the 4th column
                            R_Height = float(values[4])  # Height in the 5th column
                            if L_Height<max_recordable_height: L_vals.append(L_Height)
                            if R_Height<max_recordable_height: R_vals.append(R_Height)
                    except:
                        pass
    except KeyboardInterrupt:
        print('Stopped by user')

    serialPort.close()
    if len(L_vals)>5: L_Height = statistics.mean(L_vals)
    else: L_Height = 999
    if len(R_vals)>5: R_Height = statistics.mean(R_vals)
    else: R_Height = 999

    return affine_transform(L_Height,'L',max_recordable_height), affine_transform(R_Height,'R',max_recordable_height)

"""
Quick (1 second) height read (used for control, can  be made faster as necessary)
"""
def read_current_height(serial_port, baud_rate=115200, max_recordable_height=400):
    L_h, R_h = read_height_for(1, serial_port, baud_rate, max_recordable_height)
    return L_h, R_h

"""
Reset the ESP32 microcontroller by sending a reset command.
param serial_port: The serial port to which the test rig controller is connected.
param baud_rate: The baud rate of the serial connection.
No return.
"""
def ESP_Hard_Reset(serial_port, baud_rate=115200, loop_count=0):
    # Reset ESP
    send_character(serial_port, baud_rate, RESET_ESP)

    # Wait for ESP to reset, look for available character
    serialPort = serial.Serial(
        port=serial_port, baudrate=baud_rate, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
    )
    start_time = time.time()
    try:
        while(time.time()-start_time < 15): # give max 15 seconds for reset
            if serialPort.in_waiting > 0: # break once we have contact
                break
    except KeyboardInterrupt:
        print('Stopped by user')  
    serialPort.close()  
    time.sleep(0.5)

    L,R = read_current_height(serial_port, baud_rate)
    # R as primary
    if (R > 800) and loop_count < 2:
        loop_count += 1
        ESP_Hard_Reset(serial_port, baud_rate, loop_count)

"""
Reset the ToF sensors by sending power cycle commands.
param serial_port: The serial port to which the test rig controller is connected.
param baud_rate: The baud rate of the serial connection.
No return.
"""
def Sensor_Hard_Reset(serial_port, baud_rate=115200, loop_count=0):
    send_character(serial_port, baud_rate, SENSOR_POWER_OFF)
    time.sleep(2)
    send_character(serial_port, baud_rate, SENSOR_POWER_ON)
    ESP_Hard_Reset(serial_port, baud_rate)

    L,R = read_current_height(serial_port, baud_rate)
    # R as primary
    if (R > 800) and loop_count < 2:
        loop_count += 1
        Sensor_Hard_Reset(serial_port, baud_rate, loop_count)

"""
Reset the ToF sensors and the ESP32 microcontroller until height readings are within the recordable range.
"""
def ToF_Reset(serial_port, baud_rate=115200, max_recordable_height=400):
    ESP_Hard_Reset(serial_port, baud_rate)
    time.sleep(0.1)
    L_Height, R_Height = read_current_height(serial_port, baud_rate)
    if R_Height > max_recordable_height:
        Sensor_Hard_Reset(serial_port, baud_rate)
        time.sleep(0.1)

"""
Safety stop for the test setup - stop pump & linear actuators, release air pressure.
"""
def interrupt_stop(serial_port,  baud_rate=115200):
    send_character(serial_port, baud_rate, BOTH_STOP)
    time.sleep(0.25)
    send_character(serial_port, baud_rate, AIR_PUMP_OFF)
    time.sleep(0.25)
    send_character(serial_port, baud_rate, AIR_RELEASE_ON)
    # cap.release() # This is a good idea, but I don't think it will work, as the cap is on a seprate thread
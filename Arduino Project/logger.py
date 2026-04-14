import serial
import time
import subprocess

# --- CONFIGURATION ---
COM_PORT = 'COM3'  # Make sure this matches your Arduino port!
BAUD_RATE = 9600
FILE_NAME = "fluid_test_data.csv"

try:
    print(f"Connecting to Arduino on {COM_PORT}...")
    arduino = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)

    with open(FILE_NAME, "w") as file:
        print(f"\n--- RECORDING LIVE TO '{FILE_NAME}' ---")
        print("Press 'Ctrl + C' in this terminal to STOP and GENERATE GRAPHS.\n")

        arduino.write(b'r')

        while True:
            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8').strip()

                # 1. Print everything to the screen so you can see it
                print(line)

                # 2. THE FIX: Only write to the CSV file if the line has a comma
                if "," in line:
                    file.write(line + "\n")
                    file.flush()

except KeyboardInterrupt:
    print("\n\n--- STOP SIGNAL RECEIVED ---")

    # Safely shut down the Arduino
    arduino.write(b's')
    arduino.close()
    print(f"Success! Clean data perfectly saved to {FILE_NAME}.")

    # Automatically launch the graphing script
    print("\n--- GENERATING GRAPHS ---")
    try:
        subprocess.run(["python", "orifice_analysis.py"], check=True)
    except Exception as e:
        print(f"\nError running the analysis script: {e}")
        print("Please ensure 'orifice_analysis.py' is in the exact same folder as this script!")

except Exception as e:
    print(f"An error occurred: {e}")
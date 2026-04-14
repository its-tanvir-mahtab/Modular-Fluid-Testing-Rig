#include <OneWire.h>
#include <DallasTemperature.h>

// --- PIN ASSIGNMENTS ---
const int flowPin = 2;        
const int tempPin = 4;        
const int pressurePin = A0;   

// --- SENSOR SETUP ---
OneWire oneWire(tempPin);
DallasTemperature tempSensor(&oneWire);

volatile int pulseCount = 0; 
float flowRate = 0.0;
const float calibrationFactor = 7.9; 
unsigned long oldTime = 0;

// --- RECORDING CONTROL ---
boolean isRecording = false; // Starts paused!

void pulseCounter() {
  pulseCount++;
}

void setup() {
  Serial.begin(9600);
  
  pinMode(flowPin, INPUT); 
  attachInterrupt(digitalPinToInterrupt(flowPin), pulseCounter, FALLING);
  tempSensor.begin();
  
  // Instructions for the user
  Serial.println("RIG READY.");
  Serial.println("Type 'r' to START recording data.");
  Serial.println("Type 's' to STOP recording data.");
}

void loop() {
  
  // --- LISTEN FOR KEYBOARD COMMANDS ---
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == 'r' && isRecording == false) {
      isRecording = true;
      Serial.println("--- RECORDING STARTED ---");
      Serial.println("Flow_Rate(L/min), Diff_Pressure(Pa), Temp(C)"); // CSV Header
      oldTime = millis(); // Reset timer so we get exactly 1 second
      pulseCount = 0;     // Clear any accidental bumps to the flow sensor
    } 
    else if (command == 's' && isRecording == true) {
      isRecording = false;
      Serial.println("--- RECORDING STOPPED ---");
    }
  }

  // --- ONLY TAKE READINGS IF RECORDING IS TRUE ---
  if (isRecording == true) {
    
    // Execute exactly once every 1000 milliseconds (1 second)
    if ((millis() - oldTime) >= 1000) { 
      
      // 1. FLOW
      detachInterrupt(digitalPinToInterrupt(flowPin));
      flowRate = ((1000.0 / (millis() - oldTime)) * pulseCount) / calibrationFactor;
      pulseCount = 0; 
      oldTime = millis(); 
      attachInterrupt(digitalPinToInterrupt(flowPin), pulseCounter, FALLING);

      // 2. PRESSURE
      int rawPressure = analogRead(pressurePin);
      float voltageRatio = (float)rawPressure / 1023.0;
      float pressureKpa = (voltageRatio - 0.04) / 0.09;
      float pressurePa = pressureKpa * 1000.0;

      // 3. TEMPERATURE
      tempSensor.requestTemperatures(); 
      float tempC = tempSensor.getTempCByIndex(0);

      // 4. PRINT CSV FORMAT
      Serial.print(flowRate);
      Serial.print(",");
      Serial.print(pressurePa);
      Serial.print(",");
      Serial.println(tempC); 
    }
  }
}
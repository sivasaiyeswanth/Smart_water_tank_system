#include <SPI.h>
#include <LoRa.h>

// Define the pins used by the transceiver module
#define ss 5
#define rst 14
#define dio0 2

// Define the water level sensor pins (connected to 25%, 50%, and 75% levels)
#define WATER_SENSOR_PIN_25 25  // Pin for 25% water level
#define WATER_SENSOR_PIN_50 26  // Pin for 50% water level
#define WATER_SENSOR_PIN_75 27  // Pin for 75% water level
#define WATER_SENSOR_PIN_100 15  // Pin for 100% water level
void setup() {
  // Initialize Serial Monitor
  Serial.begin(115200);
  while (!Serial);
  Serial.println("LoRa Water Level Sender");

  // Setup water sensor pins
  pinMode(WATER_SENSOR_PIN_25, INPUT_PULLUP); // 25% level sensor
  pinMode(WATER_SENSOR_PIN_50, INPUT_PULLUP); // 50% level sensor
  pinMode(WATER_SENSOR_PIN_75, INPUT_PULLUP); // 75% level sensor
  pinMode(WATER_SENSOR_PIN_100, INPUT_PULLUP); // 100% level sensor

  // Setup LoRa transceiver module
  LoRa.setPins(ss, rst, dio0);

  // Replace the LoRa.begin(---E-) argument with your location's frequency
  while (!LoRa.begin(433E6)) {
    Serial.println("LoRa initialization failed, retrying...");
    delay(500);
  }
  
  // Change sync word (0xF3) to match the receiver
  LoRa.setSyncWord(0xF3);
  Serial.println("LoRa Initializing OK!");
}

void loop() {
  // Read water level sensor states
  int waterLevel25 = digitalRead(WATER_SENSOR_PIN_25);
  int waterLevel50 = digitalRead(WATER_SENSOR_PIN_50);
  int waterLevel75 = digitalRead(WATER_SENSOR_PIN_75);
  int waterLevel100 = digitalRead(WATER_SENSOR_PIN_100);
  String message = "";
  if (waterLevel100 == LOW ) {
    // If 50% level is reached (priority), send "50"
    message = "100"; 
    Serial.println("Water level at 100% - Sending 100");
  } 
  // Check water levels and build the message accordingly
  else if (waterLevel75 == LOW ) {
    // If 50% level is reached (priority), send "50"
    message = "75"; 
    Serial.println("Water level at 75% - Sending 75");
  } 
  else if (waterLevel50 == LOW) {
    // If only 75% level is reached, send "75"
    message = "50"; 
    Serial.println("Water level at 50% - Sending 50");
  } 
  else if (waterLevel25 == LOW) {
    // If 25% level is reached, send "25"
    message = "25"; 
    Serial.println("Water level at 25% - Sending 25");
  } 
  else {
    // No water level detected (all circuits open), send "0"
    message = "0"; 
    Serial.println("Water level low - Sending 0");
  }

  // Send the message over LoRa
  LoRa.beginPacket();
  LoRa.print(message);
  LoRa.endPacket();
  
  delay(1000);  // Wait for 1 second before sending the next packet
}

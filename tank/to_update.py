import serial
import psycopg2

# Serial port settings
serial_port = 'COM4'  # Adjust this as needed
baud_rate = 115200

# Database connection settings
db_config = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "s",
    "host": "localhost",
    "port": "5432",
}

def update_database(water_level):
    try:
        # Establish connection to PostgreSQL
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        # Update the current level in the database
        cursor.execute("UPDATE app_tankdata SET current_level = %s WHERE id = 3", (water_level,))
        conn.commit()
        print(f"Database updated with water level: {water_level}")
    except Exception as e:
        print("Error updating the database:", e)
    finally:
        cursor.close()
        conn.close()

def main():
    # Setup the serial connection
    try:
        ser = serial.Serial(serial_port, baud_rate)
        print("Listening for water level data...")

        while True:
            # Read line from serial input
            if ser.in_waiting > 0:
                try:
                    received_data = ser.readline().decode('utf-8').strip()  # Trying to decode data

                    print("Received data:", received_data)

                    # Check for water level message and parse it
                    if 'Received message' in received_data:
                        # Extract the number after 'Received message:'
                        water_level = int(received_data.split(":")[1].strip())
                        print(f"Water level received: {water_level}%")
                        update_database(water_level)

                    # Example: "Water level is at 25%"
                    elif 'Water level is at' in received_data:
                        # Extract the percentage from the sentence
                        water_level = int(received_data.split(' ')[4].replace('%', ''))
                        print(f"Water level received: {water_level}%")
                        update_database(water_level)

                except ValueError:
                    # Non-numeric data, ignore and continue
                    print("Non-numeric data received, ignoring:", received_data)

                except UnicodeDecodeError as e:
                    # Handle decoding errors (non-UTF-8 data)
                    print(f"UnicodeDecodeError: {e}")
                    # Optionally, print raw bytes
                    raw_data = ser.readline()
                    print(f"Raw data (bytes): {raw_data}")

    except serial.SerialException as e:
        print("Error opening serial port:", e)

if __name__ == "__main__":
    main()

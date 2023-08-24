import requests
import time
import csv
from datetime import datetime

# Constants
URL = "http://0.0.0.0:5000/chat"
NUM_REQUESTS = 1000
input_payload = {
    "input_data": {
        "input_string": "Hello, how are you?"
    },
    "parameters": {
        "temperature": 0.6,
        "top_p": 0.9,
        "max_gen_len": 64
    }
}

# Generate a unique run_id based on the current timestamp
run_id = int(time.time())

# List to store latencies
latencies = []

# Open the CSV for writing
with open('requests.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    # writer.writerow(["Run ID", "Request ID", "Request #", "Latency (ms)", "Date"])

    for i in range(NUM_REQUESTS):
        start_time = time.time()
        
        response = requests.post(URL, json=input_payload)
        
        # Get the date/time for this request
        request_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if response.status_code == 200:
            elapsed_time = (time.time() - start_time) * 1000
            print(f"Request #{i+1} elapsed time", elapsed_time)
            latencies.append(elapsed_time)
            request_id = f"{run_id}-{i+1}"
            writer.writerow([run_id, request_id, i+1, elapsed_time, request_date])
        else:
            print(f"Request #{i+1} failed with status code {response.status_code}. Error: {response.text}")

# Calculate statistics
average_latency = sum(latencies) / len(latencies)
max_latency = max(latencies)
min_latency = min(latencies)

print(f"Average latency: {average_latency:.2f} ms")
print(f"Max latency: {max_latency:.2f} ms")
print(f"Min latency: {min_latency:.2f} ms")

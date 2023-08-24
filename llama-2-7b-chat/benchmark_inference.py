import requests
import time

# Constants
URL = "http://0.0.0.0:5000/chat" # Needs to be updated based on external IP
NUM_REQUESTS = 1000  # Adjust the number of requests based on your requirements
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

# List to store latencies
latencies = []

for _ in range(NUM_REQUESTS):
    start_time = time.time()
    
    response = requests.post(URL, json=input_payload)

    if response.status_code == 200:
        elapsed_time = (time.time() - start_time) * 1000  # in milliseconds
        print("elapsed time", elapsed_time)
        latencies.append(elapsed_time)
    else:
        print(f"Request failed with status code {response.status_code}. Error: {response.text}")

# Calculate statistics
average_latency = sum(latencies) / len(latencies)
max_latency = max(latencies)
min_latency = min(latencies)

print(f"Average latency: {average_latency:.2f} ms")
print(f"Max latency: {max_latency:.2f} ms")
print(f"Min latency: {min_latency:.2f} ms")


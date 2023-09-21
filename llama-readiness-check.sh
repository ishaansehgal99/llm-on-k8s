#!/bin/bash

# Extract the ordinal from the pod's hostname
ordinal=$(hostname | grep -o '[^-]*$')

# Common check: Count the number of web_example_chat_completion.py processes
web_example_count=$(pgrep -af "python -u web_example_chat_completion.py" | wc -l)

# Get PID of torchrun process
torchrun_pid=$(pgrep -f "torchrun")

# Check if torchrun exists and get nproc_per_node value
if [[ -n "$torchrun_pid" ]]; then
    nproc_per_node_value=$(cat /proc/$torchrun_pid/cmdline | tr '\0' ' ' | awk -F"--nproc_per_node=" '{print $2}' | awk '{print $1}')
else
    exit 1  # readiness probe fails if torchrun doesn't exist
fi

# Check conditions for all pods
if [[ $web_example_count -ne $nproc_per_node_value ]]; then
    exit 1  # readiness probe fails if count doesn't match nproc_per_node
fi

# Additional check for pod 0
if [[ $ordinal -eq 0 ]]; then
    if ! curl -s http://localhost:5000/healthz | grep -q "Healthy"; then
        exit 1  # readiness probe fails if pod 0 health check doesn't pass
    fi
fi

exit 0  # readiness probe success

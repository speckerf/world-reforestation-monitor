#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -n <number_of_terminals> -c <command>"
    exit 1
}

# Parse the input parameters
while getopts ":n:c:" opt; do
    case $opt in
        n) num_terminals=$OPTARG ;;
        c) command_to_run=$OPTARG ;;
        *) usage ;;
    esac
done

# Check if both -n and -c options are provided
if [ -z "$num_terminals" ] || [ -z "$command_to_run" ]; then
    usage
fi

# Run the command in the first terminal
osascript -e "tell application \"Terminal\" to do script \"$command_to_run\""

# Wait for 5 seconds before starting the rest
sleep 5

# Run the command in the remaining terminals
for ((i = 2; i <= num_terminals; i++)); do
    osascript -e "tell application \"Terminal\" to do script \"$command_to_run\""
done


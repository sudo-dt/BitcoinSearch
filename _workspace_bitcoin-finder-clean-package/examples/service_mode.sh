#!/bin/bash

# Service mode example
# This script demonstrates running the tool in service mode with automatic checkpointing

# Navigate to the tool directory
cd ../bitcoin-finder-fixed

# Run the tool in service mode
./target/debug/bitcoin-finder \
  -c ../examples/service_config.json \
  -w ../examples/advanced_wordlist.txt \
  -d ../examples/sample_addresses.txt \
  --service-mode \
  --auto-resume \
  --checkpoint-interval 300
#!/bin/bash

# Advanced search example with learning features
# This script demonstrates an advanced search with learning features enabled

# Navigate to the tool directory
cd ../bitcoin-finder-fixed

# Run the tool with advanced configuration and learning mode
./target/debug/bitcoin-finder \
  -c ../examples/advanced_config.json \
  -w ../examples/advanced_wordlist.txt \
  -d ../examples/sample_addresses.txt \
  --learning-mode \
  --auto-resume
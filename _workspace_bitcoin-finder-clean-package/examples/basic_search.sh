#!/bin/bash

# Basic search example
# This script demonstrates a basic search with the Bitcoin Wallet Recovery Tool

# Navigate to the tool directory
cd ../bitcoin-finder-fixed

# Run the tool with basic configuration
./target/debug/bitcoin-finder \
  -c ../examples/basic_config.json \
  -w ../examples/basic_wordlist.txt \
  -d ../examples/sample_addresses.txt
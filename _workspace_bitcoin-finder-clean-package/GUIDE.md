# Bitcoin Wallet Recovery Tool - User Guide

## Overview

The Bitcoin Wallet Recovery Tool is a powerful utility designed to search for Bitcoin wallets that contain specific words in their mnemonics. It can be used to recover lost wallets or to find wallets that match specific criteria.

## Features

- **Custom Word List Filtering**: Search for mnemonics containing specific words from a custom word list
- **Multiple Derivation Paths**: Supports various Bitcoin address derivation paths (BIP44, BIP49, BIP84)
- **Address Types**: Generates and checks Legacy (P2PKH), SegWit (P2SH-P2WPKH), Native SegWit (P2WPKH), and Taproot (P2TR) addresses
- **Learning System**: Adaptive search that learns from patterns to improve search efficiency
- **Checkpoint System**: Save and resume searches from checkpoints
- **Health Monitoring**: Monitors system resources and performance
- **Service Mode**: Run as a continuous service with automatic checkpointing

## Installation

### Prerequisites

- Rust and Cargo (1.63.0 or later)
- Build essentials (for compiling dependencies)

### Building from Source

1. Clone the repository or extract the source code
2. Navigate to the project directory
3. Build the project using Cargo:

```bash
cargo build --release
```

4. The binary will be available in `target/release/bitcoin-finder`

## Configuration

The tool uses a JSON configuration file to customize its behavior. Here's an example configuration:

```json
{
  "min_matching_words": 1,
  "max_addresses_per_path": 20,
  "highlight_threshold": 3,
  "word_match_logging": true,
  "learning_enabled": true,
  "min_score_threshold": 0.5,
  "max_high_scores": 100,
  "adaptive_search": true,
  "word_weight_multiplier": 1.5,
  "pattern_recognition": true,
  "position_aware_matching": true,
  "health_check_interval": 60,
  "max_memory_percent": 90,
  "max_checkpoint_size_mb": 100,
  "checkpoint_compression": true,
  "paths_to_check": [
    "m/44'/0'/0'/0",
    "m/44'/0'/0'/1",
    "m/44'/0'/1'/0",
    "m/44'/0'/1'/1",
    "m/49'/0'/0'/0",
    "m/49'/0'/0'/1",
    "m/49'/0'/1'/0",
    "m/49'/0'/1'/1",
    "m/84'/0'/0'/0",
    "m/84'/0'/0'/1",
    "m/84'/0'/1'/0",
    "m/84'/0'/1'/1"
  ]
}
```

### Configuration Options

- **min_matching_words**: Minimum number of words from the word list that must be present in a mnemonic to be considered a match
- **max_addresses_per_path**: Maximum number of addresses to generate per derivation path
- **highlight_threshold**: Minimum number of matching words to highlight in the output
- **word_match_logging**: Whether to log mnemonics with matching words
- **learning_enabled**: Enable the learning system to improve search efficiency
- **min_score_threshold**: Minimum score for a mnemonic to be considered by the learning system
- **max_high_scores**: Maximum number of high scores to keep in the learning system
- **adaptive_search**: Enable adaptive search based on learning
- **word_weight_multiplier**: Multiplier for word weights in the learning system
- **pattern_recognition**: Enable pattern recognition in the learning system
- **position_aware_matching**: Consider word positions in the learning system
- **health_check_interval**: Interval in seconds between health checks
- **max_memory_percent**: Maximum memory usage percentage before triggering a health warning
- **max_checkpoint_size_mb**: Maximum checkpoint file size in MB
- **checkpoint_compression**: Whether to compress checkpoint files
- **paths_to_check**: List of derivation paths to check

## Required Files

1. **Word List File**: A text file containing words to search for in mnemonics (one word per line)
2. **Address Database File**: A text file containing Bitcoin addresses to check against (one address per line)
3. **Configuration File**: A JSON file with the tool's configuration

## Usage

Basic usage:

```bash
./bitcoin-finder -c config.json -w wordlist.txt -d btcdatabase.txt
```

With auto-resume from checkpoint:

```bash
./bitcoin-finder -c config.json -w wordlist.txt -d btcdatabase.txt --auto-resume
```

Running in service mode:

```bash
./bitcoin-finder -c config.json -w wordlist.txt -d btcdatabase.txt --service-mode
```

With enhanced learning mode:

```bash
./bitcoin-finder -c config.json -w wordlist.txt -d btcdatabase.txt --learning-mode
```

### Command Line Options

- `-c, --config <FILE>`: Path to config file (default: config.json)
- `-w, --wordlist <FILE>`: Path to word list file (default: wordlist.txt)
- `-d, --database <FILE>`: Path to address database file (default: btcdatabase.txt)
- `--service-mode`: Run in service mode with automatic checkpointing and recovery
- `--auto-resume`: Automatically resume from last checkpoint if available
- `--checkpoint-interval <SECONDS>`: Interval in seconds between checkpoints (default: 900)
- `-a, --adaptive`: Enable adaptive search mode
- `-l, --learning-mode`: Run in enhanced learning mode

## How It Works

1. The tool generates random BIP39 mnemonics
2. It checks if the generated mnemonics contain words from the custom word list
3. If a mnemonic contains enough matching words, it derives Bitcoin addresses from it
4. It checks the derived addresses against the address database
5. If a match is found, it saves the mnemonic, derivation path, and address to a file
6. The learning system analyzes patterns in successful matches to improve search efficiency

## Output Files

- **logs/**: Contains log files with detailed information about the search process
- **checkpoints/**: Contains checkpoint files for resuming searches
- **models/**: Contains learning system models
- **matches/**: Contains information about found matches

## Performance Considerations

- The tool is CPU-intensive and can use multiple threads
- Memory usage depends on the size of the address database
- Checkpoint files can grow large if the search runs for a long time
- The learning system can improve search efficiency over time

## Troubleshooting

- If the tool crashes, try running with `--auto-resume` to resume from the last checkpoint
- If memory usage is too high, reduce `max_addresses_per_path` or the number of paths in the configuration
- If the tool is too slow, try running with `--learning-mode` to improve search efficiency
- If checkpoint files are too large, reduce `checkpoint_interval` or disable checkpointing

## Security Considerations

- The tool generates and processes private keys, so it should be run in a secure environment
- Checkpoint files contain sensitive information and should be protected
- The address database should be kept secure to prevent unauthorized access
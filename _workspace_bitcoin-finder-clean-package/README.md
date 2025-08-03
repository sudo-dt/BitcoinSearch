# Bitcoin Wallet Recovery Tool

A high-performance tool for finding Bitcoin wallets with specific word patterns in their mnemonics.

## Features

- **Custom Word Filtering**: Search for mnemonics containing specific words from your custom list
- **Multi-Path Support**: Checks multiple derivation paths (BIP44, BIP49, BIP84)
- **Address Types**: Supports Legacy, SegWit, Native SegWit, and Taproot addresses
- **Adaptive Learning**: Improves search efficiency over time by learning from patterns
- **Checkpoint System**: Save and resume searches from checkpoints
- **Performance Optimized**: Multi-threaded design with memory-mapped database access

## Quick Start

1. Build the tool:
   ```bash
   cargo build --release
   ```

2. Prepare your files:
   - `wordlist.txt`: Words to search for in mnemonics (one per line)
   - `btcdatabase.txt`: Bitcoin addresses to check against (one per line)
   - `config.json`: Configuration settings (see example in repo)

3. Run the tool:
   ```bash
   ./bitcoin-finder -c config.json -w wordlist.txt -d btcdatabase.txt
   ```

## Configuration

The tool is highly configurable through the `config.json` file:

```json
{
  "min_matching_words": 1,
  "max_addresses_per_path": 20,
  "highlight_threshold": 3,
  "word_match_logging": true,
  "learning_enabled": true,
  "adaptive_search": true,
  "paths_to_check": [
    "m/44'/0'/0'/0",
    "m/49'/0'/0'/0",
    "m/84'/0'/0'/0"
  ]
}
```

## Command Line Options

- `-c, --config <FILE>`: Path to config file
- `-w, --wordlist <FILE>`: Path to word list file
- `-d, --database <FILE>`: Path to address database file
- `--service-mode`: Run as a continuous service
- `--auto-resume`: Resume from last checkpoint
- `-l, --learning-mode`: Enable enhanced learning features

## How It Works

1. Generates random BIP39 mnemonics
2. Checks if mnemonics contain words from your custom list
3. Derives Bitcoin addresses from matching mnemonics
4. Checks addresses against your database
5. Saves matches with private keys for recovery

## Performance Tips

- Use `--learning-mode` for improved search efficiency
- Adjust `max_addresses_per_path` based on your system's memory
- Use `--service-mode` for long-running searches
- Enable checkpointing for resumable searches

## For More Information

See the [GUIDE.md](GUIDE.md) file for detailed usage instructions and configuration options.
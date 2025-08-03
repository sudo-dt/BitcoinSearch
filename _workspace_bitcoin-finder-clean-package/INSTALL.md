# Bitcoin Wallet Recovery Tool - Installation Guide

## Prerequisites

- Rust and Cargo (1.63.0 or later)
- Build essentials (for compiling dependencies)
- Linux, macOS, or Windows with WSL

## Installation Steps

1. **Extract the Source Code**

   ```bash
   unzip bitcoin-finder-fixed-v3.zip
   cd bitcoin-finder-fixed
   ```

2. **Build the Tool**

   ```bash
   cargo build --release
   ```

   This will create the binary at `target/release/bitcoin-finder`

3. **Prepare Your Files**

   - Create or modify `config.json` with your desired settings
   - Create `wordlist.txt` with the words you want to search for
   - Create `btcdatabase.txt` with the Bitcoin addresses to check against

   Example files are provided in the `examples` directory.

4. **Run the Tool**

   ```bash
   ./target/release/bitcoin-finder -c config.json -w wordlist.txt -d btcdatabase.txt
   ```

   For more options, run:

   ```bash
   ./target/release/bitcoin-finder --help
   ```

## Example Usage

The `examples` directory contains several example scripts and configuration files:

- `basic_search.sh`: Simple search with minimal configuration
- `advanced_search.sh`: Advanced search with learning features
- `service_mode.sh`: Running as a service with automatic checkpointing

To run an example:

```bash
cd examples
./basic_search.sh
```

## Troubleshooting

- If you encounter compilation errors, make sure you have the latest version of Rust and Cargo
- If the tool crashes, try running with `--auto-resume` to resume from the last checkpoint
- If memory usage is too high, reduce `max_addresses_per_path` or the number of paths in the configuration

## Additional Resources

- See `README.md` for a quick overview
- See `GUIDE.md` for detailed usage instructions
- See `SUMMARY.md` for a project summary
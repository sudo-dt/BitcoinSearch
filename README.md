# Bitcoin Search

A high-performance tool for finding Bitcoin wallets with specific word patterns in their mnemonics.

## Repository Structure

This repository contains two implementations:

1. **Original Implementation** - Located in `_workspace_bitcoin-finder-clean-package/`
   - The original Bitcoin seed phrase finder with word matching reports

2. **Optimized Implementation** - Located in `optimized/`
   - An optimized version that removes unnecessary matching reports
   - Focuses on core functionality with improved performance
   - Includes machine learning capabilities for better search efficiency
   - Supports personalized word lists for targeted searches

## Optimized Implementation Features

- **Efficient Bruteforcing**: Generates and tests Bitcoin seed phrases at high speed
- **Machine Learning Integration**: Uses adaptive learning to improve search efficiency
- **Personalized Word Lists**: Supports custom word lists for targeted searches
- **Multi-threaded**: Utilizes all available CPU cores for maximum performance
- **Checkpoint System**: Automatically saves progress and can resume from checkpoints
- **Health Monitoring**: Monitors system resources and performance

## Getting Started

See the README.md files in each implementation directory for specific instructions:

- [Original Implementation](./_workspace_bitcoin-finder-clean-package/README.md)
- [Optimized Implementation](./optimized/README.md)
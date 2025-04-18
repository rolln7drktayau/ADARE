# ADARE Environment Setup Guide

This guide will help you set up the necessary environment to run the ADARE vs NSGA-III comparison.

## Prerequisites

- Python 3.8 or higher
- Git (to clone the repository)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/rolln7drktayau/ADARE.git
cd ADARE
```

### 2. Set Up the Environment

You can use the provided setup script:

```bash
chmod +x ADARE/utils/setup.sh
./ADARE/utils/setup.sh
```

Or follow these manual steps:

```bash
# Create a virtual environment
python -m venv adare_env

# Activate the virtual environment

# On Windows:
adare_env\Scripts\activate

# On macOS/Linux:
source adare_env/bin/activate

# Install dependencies
pip install -r ADARE/utils/requirements.txt
```

### 3. Verify Data Files

Ensure the following files exist in the `data` directory:
- `algorithm_config.json`
- `environments.json`
- `tasks.json`

### 4. Run the Comparison

```bash
python ADARE/adare_vs_nsga3.py
```

## Troubleshooting

If you encounter any issues with the dependencies:

1. Make sure you're using Python 3.8+
2. Try installing dependencies individually:
   ```bash
   pip install numpy scipy matplotlib deap pymoo g4f
   ```
3. If g4f installation fails, you can install it from source:
   ```bash
   pip install git+https://github.com/xtekky/gpt4free.git
   ```

## Output

After running the script, you'll find:
- Visualization plots in the `output/plots` directory
- Comparison reports in the `output/reports` directory

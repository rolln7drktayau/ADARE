# Variables
PYTHON = python3
VENV = myenv
PIP = $(VENV)/bin/pip
REQUIREMENTS = requirements.txt

# Directories
DATA_DIR = data
HISTORY_DIR = $(DATA_DIR)/history
BENCHMARKS_DIR = $(DATA_DIR)/benchmarks
OUTPUT_DIR = output
WORKFLOWS_DIR = $(DATA_DIR)/workflows
BUILD_DIR = $(DATA_DIR)/build

# Main commands
.PHONY: all run clean build test setup

all: setup run

# Default run with sample data
run:
	@mkdir -p $(OUTPUT_DIR)/plots $(OUTPUT_DIR)/reports
	$(PYTHON) adare_vs_nsga3.py

# Run with specific workflow
run-%:
	@workflow_type=$$(echo $* | cut -d'_' -f1); \
	workflow_size=$$(echo $* | cut -d'_' -f2); \
	echo "Processing $$workflow_type workflow with size $$workflow_size"; \
	mkdir -p $(HISTORY_DIR) $(BENCHMARKS_DIR)/$$workflow_type $(OUTPUT_DIR)/plots $(OUTPUT_DIR)/reports; \
	target_json="$(BENCHMARKS_DIR)/$$workflow_type/$*.json"; \
	if [ -f "$$target_json" ]; then \
		echo "Benchmark file already exists at $$target_json. Skipping conversion."; \
	else \
		echo "Benchmark file not found. Converting from XML..."; \
		xml_path="$(WORKFLOWS_DIR)/$$workflow_type/$*.xml"; \
		if [ ! -f "$$xml_path" ]; then \
			echo "Error: XML file not found at $$xml_path"; \
			exit 1; \
		fi; \
		$(PYTHON) $(BUILD_DIR)/xml_to_json.py "$$xml_path" -o $(HISTORY_DIR)/$*.json && \
		$(PYTHON) $(BUILD_DIR)/format.py $(HISTORY_DIR)/$*.json -o "$$target_json" && \
		echo "Conversion completed successfully."; \
	fi; \
	$(PYTHON) adare_vs_nsga3.py $*

clean:
	rm -rf $(OUTPUT_DIR)/*
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.pyc
	rm -rf $(HISTORY_DIR)/*

test:
	$(PYTHON) -m pytest tests/

setup: $(VENV)/bin/activate
	$(PIP) install -r $(REQUIREMENTS)

# Virtual environment
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

# Additional utilities
.PHONY: format lint help

format:
	$(PYTHON) -m black .

lint:
	$(PYTHON) -m flake8 .

help:
	@echo "Available commands:"
	@echo "  make run              - Run the ADARE algorithm with sample data"
	@echo "  make run-WORKFLOW_SIZE - Run with specific workflow (e.g., make run-CyberShake_30)"
	@echo "  make clean            - Remove generated files"
	@echo "  make test             - Run tests"
	@echo "  make setup            - Install dependencies"
	@echo "  make format           - Format code with black"
	@echo "  make lint             - Check code style with flake8"
	@echo "  make help             - Display this help message"
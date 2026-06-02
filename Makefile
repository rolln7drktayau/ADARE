# Variables
PYTHON ?= python
VENV = myenv
REQUIREMENTS = requirements.txt

ifeq ($(OS),Windows_NT)
	VENV_ACTIVATE = $(VENV)/Scripts/activate
	PIP = $(VENV)/Scripts/pip.exe
else
	VENV_ACTIVATE = $(VENV)/bin/activate
	PIP = $(VENV)/bin/pip
endif

# Directories
DATA_DIR = data
HISTORY_DIR = $(DATA_DIR)/history
BENCHMARKS_DIR = $(DATA_DIR)/benchmarks
OUTPUT_DIR = output
WORKFLOWS_DIR = $(DATA_DIR)/workflows
BUILD_DIR = $(DATA_DIR)/build
COOKIES_DIR = har_and_cookies

# Main commands
.PHONY: all menu run clean build setup format lint help smoke main20 extended-small extended-1000-r20 extended-3000-r20 ablation paper

all: menu

menu:
	$(PYTHON) scripts/make_menu.py

# Default run with sample data
run:
	@mkdir -p $(OUTPUT_DIR)/plots $(OUTPUT_DIR)/reports
	$(PYTHON) main.py

smoke:
	$(PYTHON) main.py --benchmarks Montage_25 --runs 1 --generations 5 --population-size 30

main20:
	$(PYTHON) main.py --benchmarks Montage_25 CyberShake_30 Epigenomics_24 --runs 20 --generations 70 --population-size 100

extended-small:
	$(PYTHON) run_extended_comparison.py --benchmarks Montage_25 CyberShake_30 Epigenomics_24 --algorithms ADARE NSGA-III NSGA-II MOEA/D QL-NSGA-III OVEA-style QMOEA/D-AWA-style --runs 5 --generations 15 --population-size 80 --output-dir output/extended_small_menu

extended-1000-r20:
	$(PYTHON) run_extended_comparison.py --benchmarks CyberShake_1000 Inspiral_1000 Montage_1000 Sipht_1000 --algorithms ADARE NSGA-III QL-NSGA-III OVEA-style QMOEA/D-AWA-style --runs 20 --generations 15 --population-size 80 --output-dir output/extended_1000_r20

extended-3000-r20:
	$(PYTHON) run_extended_comparison.py --benchmarks Montage_3000_wfcommons Epigenomics_3000_wfcommons Seismology_3000_wfcommons Soykb_3000_wfcommons Srasearch_3000_wfcommons --algorithms ADARE NSGA-III QL-NSGA-III OVEA-style QMOEA/D-AWA-style --runs 20 --generations 8 --population-size 60 --output-dir output/extended_3000_r20

ablation:
	$(PYTHON) run_ablation_v1_v5.py --runs 20 --output-dir output/ablation_full

paper:
	$(PYTHON) -c "import shutil; shutil.copyfile('article_ecml.tex','papers/article_ecml.tex')"
	pdflatex -interaction=nonstopmode article_ecml.tex
	pdflatex -interaction=nonstopmode article_ecml.tex
	$(PYTHON) -c "import shutil; shutil.copyfile('article_ecml.pdf','papers/article_ecml.pdf'); shutil.copyfile('article_ecml.pdf','ADARE_Adaptive_Data-driven_Algorithm_for_Resource_Evolution.pdf')"

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
	rm -rf $(COOKIES_DIR)/*

setup: $(VENV_ACTIVATE)
	$(PIP) install -r $(REQUIREMENTS)

# Virtual environment
$(VENV_ACTIVATE):
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
	@echo "  make / make menu      - Open the interactive ADARE menu with estimated durations"
	@echo "  make run              - Run the ADARE algorithm with sample data"
	@echo "  make smoke            - Quick sanity run (~1-3 min)"
	@echo "  make main20           - Main 20-run paper protocol (~45-90 min)"
	@echo "  make extended-small   - Extended small-suite comparison (~20-45 min)"
	@echo "  make extended-1000-r20 - Long 1000-task 20-run protocol (~2h-2h15)"
	@echo "  make extended-3000-r20 - Long 3000-task 20-run protocol (~3h-4h)"
	@echo "  make ablation         - V1-V5 ablation (~30-75 min)"
	@echo "  make paper            - Compile and sync PDFs (~10-30 sec)"
	@echo "  make run-WORKFLOW_SIZE - Run with specific workflow (e.g., make run-CyberShake_30)"
	@echo "  make clean            - Remove generated files"
	@echo "  make setup            - Install dependencies"
	@echo "  make format           - Format code with black"
	@echo "  make lint             - Check code style with flake8"
	@echo "  make help             - Display this help message"

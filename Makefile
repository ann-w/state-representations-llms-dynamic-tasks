# Makefile for State Representations in LLMs for Dynamic Tasks
# 
# This Makefile provides convenient commands for setting up and running
# experiments on both SmartPlay and BALROG benchmarks.

.PHONY: help install-smartplay install-balrog install-all test clean

# Default target - show help
help:
	@echo "State Representations in LLMs for Dynamic Tasks"
	@echo "================================================"
	@echo ""
	@echo "Installation:"
	@echo "  make install-smartplay    Install SmartPlay benchmark environment"
	@echo "  make install-balrog       Install BALROG benchmark environment"
	@echo "  make install-all          Install both environments"
	@echo ""
	@echo "Testing:"
	@echo "  make test                 Run all SmartPlay tests"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                Remove Python cache files"
	@echo "  make clean-envs           Remove conda environments"
	@echo ""
	@echo "Usage Examples:"
	@echo "  After installation, activate the environment you want to use:"
	@echo "    conda activate smartplay"
	@echo "    conda activate balrog"

# Install SmartPlay benchmark
install-smartplay:
	@echo "Installing SmartPlay benchmark..."
	@chmod +x scripts/setup_smartplay.sh
	@./scripts/setup_smartplay.sh

# Install BALROG benchmark
install-balrog:
	@echo "Installing BALROG benchmark..."
	@chmod +x scripts/setup_balrog.sh
	@./scripts/setup_balrog.sh

# Install both benchmarks
install-all: install-smartplay install-balrog
	@echo ""
	@echo "Both benchmarks installed successfully!"
	@echo "Use 'conda activate smartplay' or 'conda activate balrog' to switch between them."

# Run tests (SmartPlay)
test:
	@echo "Running SmartPlay tests..."
	@cd smartplay && \
		source "$$(conda info --base)/etc/profile.d/conda.sh" && \
		conda activate smartplay && \
		export PYTHONPATH=$$PYTHONPATH:$$(pwd)/src && \
		python -m pytest tests/ -v

# Clean Python cache files
clean:
	@echo "Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Done!"

# Remove conda environments
clean-envs:
	@echo "Removing conda environments..."
	@conda env remove -n smartplay -y 2>/dev/null || true
	@conda env remove -n balrog -y 2>/dev/null || true
	@echo "Done!"

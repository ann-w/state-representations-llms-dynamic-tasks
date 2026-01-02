# State Representations for LLMs in Dynamic Reasoning Tasks

This repository contains the code for the paper **"State Representations for Large Language Models in Dynamic Reasoning Tasks"**.

## Abstract

As large language models move from static reasoning tasks toward dynamic environments, their success depends on the ability to navigate and respond to an environment that changes as they interact with it. An underexplored factor in these settings is the inference-time state representation. Holding model parameters fixed, we systematically vary state granularity (long form versus summary), structure (natural language versus symbolic), and spatial grounding (text-only versus images or textual map encodings) across sequential decision-making benchmarks. We find that trajectory summarisation improves performance by reducing noise and stabilising long-horizon reasoning, provided it retains the details needed to solve the task. Second, natural language representations are more effective than structured state representations. Gains are largely confined to models with code or structured output priors, such as JSON schemas. Third, while images inputs show limited benefits, text-based spatial encodings prove most effective. This advantage stems not from the spatial information itself, but from the act of construction, which compels the model to perform the spatial reasoning that static input does not elicit. Overall, we demonstrate that the design of the state interface is a decisive factor in performance, distinct from the availability of information itself. Yet even with improved representations, current large language models and vision–language models remain brittle over long horizons, particularly when they must maintain a consistent state and integrate spatial information across time.

## Repository Structure

This repository contains two separate benchmark suites, each with its own environment and dependencies:

```
state-representations-llms-dynamic-tasks/
├── smartplay/           # SmartPlay benchmark (Hanoi, Messenger, Bandits)
├── BALROG/              # BALROG benchmark (BabyAI, NetHack, TextWorld)
├── scripts/             # Installation helper scripts
├── Makefile             # Easy setup commands
└── README.md
```

### Why Two Separate Benchmarks?

The benchmarks have **incompatible dependencies** (different gym versions, different Python requirements), so they must be installed in separate conda environments. This is intentional and ensures reproducibility.

| Benchmark | Environments | Focus |
|-----------|--------------|-------|
| **SmartPlay** | Tower of Hanoi, Messenger, Multi-armed Bandits | Classical planning, state representation variations |
| **BALROG** | BabyAI, NetHack, TextWorld | Vision-language, long-horizon reasoning |

## Quick Start

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- Git
- macOS/Linux (Windows via WSL)

### Option 1: Using Make (Recommended)

```bash
# Clone the repository
git clone https://github.com/ann-w/state-representations-llms-dynamic-tasks.git
cd state-representations-llms-dynamic-tasks

# Install SmartPlay benchmark
make install-smartplay

# Install BALROG benchmark
make install-balrog

# Or install both
make install-all
```

### Option 2: Using Setup Scripts

```bash
# Install SmartPlay
./scripts/setup_smartplay.sh

# Install BALROG
./scripts/setup_balrog.sh
```

### Option 3: Manual Installation

See detailed instructions in each benchmark's README:
- [SmartPlay Installation](smartplay/README.md)
- [BALROG Installation](BALROG/README.md)

## Running Experiments

### SmartPlay Benchmark

```bash
# Activate environment
conda activate smartplay

# Navigate to smartplay directory
cd smartplay

# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run experiment
python -m scripts.main
```

**Configuration:** Edit `src/config/experiment_settings/experiment_settings.yaml` to configure:
- Model selection (`model_name`, `api_type`)
- Environment (`env_names`)
- State representation options (`use_rolling_summary`, `use_oracle_summary`, `use_visualization_of_thought`)

### BALROG Benchmark

```bash
# Activate environment
conda activate balrog

# Navigate to BALROG directory
cd BALROG

# Run evaluation
python eval.py \
  agent.type=naive \
  agent.max_text_history=16 \
  envs.names=babyai \
  client.client_name=openai \
  client.model_id=gpt-4o
```

## State Representation Options

### Trajectory Representation

| Option | Description | Config |
|--------|-------------|--------|
| **Full Trajectory** | Complete history of observations and actions | `use_rolling_summary: False` |
| **Rolling Summary** | LLM-generated summary updated each step | `use_rolling_summary: True` |
| **Oracle Summary** | Ground-truth state summary (ablation) | `use_oracle_summary: True` |

### State Structure

| Option | Description |
|--------|-------------|
| **Natural Language** | Human-readable descriptions |
| **Symbolic/Matrix** | Structured representations (JSON, matrices) |

### Spatial Grounding

| Option | Description |
|--------|-------------|
| **Text-only** | No spatial information |
| **Vision** | Image observations |
| **Visualization-of-Thought** | LLM constructs ASCII maps |
| **Oracle VoT** | Ground-truth ASCII maps (ablation) | 

## Supported Models

### SmartPlay
- OpenAI (GPT-4, GPT-4o)
- Ollama (local models)
- HuggingFace (local/API)
- DeepSeek

### BALROG
- OpenAI, Anthropic, Google Gemini (API)
- vLLM (local serving)
- HuggingFace Transformers (local)

## API Keys

Create a `.env` file in the respective benchmark directory:

```bash
# For SmartPlay (smartplay/.env)
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here

# For BALROG (BALROG/.env or export)
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```
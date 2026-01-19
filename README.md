# State Design Matters: How Representations Shape Dynamic Reasoning in Large Language Models

This repository contains the code for the paper **State Design Matters: How Representations Shape Dynamic Reasoning in Large Language Models**.

## Abstract

As large language models (LLMs) move from static reasoning tasks toward dynamic environments, their success depends on the ability to navigate and respond to an environment that changes as they interact at inference time. An underexplored factor in these settings is the representation of the state. Holding model parameters fixed, we systematically vary three key aspects: (1) state granularity (long form versus summary), (2) structure (natural language versus symbolic), and (3) spatial grounding (text-only versus images or textual map encodings) across sequential decision-making benchmarks. We find that trajectory summarisation improves performance by reducing noise and stabilising long-horizon reasoning. Second, natural language representations are more effective than structured state representations in models with code or structured output priors, such as JSON schemas. Third, while image-inputs show some benefit, text-based spatial encodings prove most effective. This advantage stems not from the spatial information itself, but from the act of construction, which compels the model to perform the spatial reasoning that static input does not elicit. Overall, we demonstrate that design choices for representing state are a decisive factor in performance, distinct from the availability of information itself. We note, however, that even with improved representations, current LLMs and VLMs remain brittle over long horizons, particularly when they must synthesise information to manage multiple subtasks to reach a goal.

## Repository Structure

This repository contains two separate benchmark suites, each with its own environment and dependencies:

```
state-representations-llms-dynamic-tasks/
├── smartplay/           # SmartPlay benchmark (Hanoi, Messenger)
├── BALROG/              # BALROG benchmark (BabyAI)
├── scripts/             
├── Makefile             # Easy setup commands
└── README.md
```

The benchmarks have **incompatible dependencies** (different gym versions, different Python requirements), so they must be installed in separate conda environments. This is intentional and ensures reproducibility.

## Quick Start

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) 

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

### Trajectory Summary Representation

| Option | Description | Config |
|--------|-------------|--------|
| **Full Trajectory** | Complete history of observations and actions | `use_rolling_summary: False` |
| **Rolling Summary** | LLM-generated summary updated each step | `use_rolling_summary: True` |
| **Oracle Summary** | Ground-truth state summary (ablation) | `use_oracle_summary: True` |

### State Structure

State structure is set through the environment name in `env_names`.

**Hanoi:**
| Representation | Environment Name |
|----------------|------------------|
| Natural Language | `Hanoi3DiskNaturalLanguage` |
| Tagged List | `Hanoi3Disk` (default) |
| Matrix | `Hanoi3DiskMatrix` |
| Dict List | `Hanoi3DiskDictList` |

**Messenger:**
| Representation | Environment Name |
|----------------|------------------|
| Natural Language | `MessengerL1` (default) |
| Natural Language + Position | `MessengerL1NaturalLanguagePos` |
| Coordinates | `MessengerL1Coordinates` |
| Symbolic | `MessengerL1Symbolic` |

### Spatial Grounding

| Option | Description | Config |
|--------|-------------|--------|
| **Text-only** | No spatial information | `use_vision: False`, `use_visualization_of_thought: False` |
| **Vision** | Image observations | `use_vision: True` |
| **Visualization-of-Thought** | LLM constructs ASCII maps | `use_visualization_of_thought: True` |
| **Oracle VoT** | Ground-truth ASCII maps (ablation) | `use_oracle_vot: True` | 

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
# Agentic Reasoning System

A Python implementation of an agentic reasoning system that builds self-organizing knowledge networks through deep graph reasoning. This project is based on the TLA+ specification found in `TLA+SPEC.md`.

## Overview

This system implements a knowledge graph-based reasoning framework that:

1. Iteratively builds a knowledge graph through reasoning
2. Uses LLMs (GPT-4o-mini and Gemini Pro) to generate reasoning and extract knowledge
3. Maintains graph-theoretic properties like scale-free networks, modularity, and bridge nodes
4. Performs compositional reasoning over the graph to identify atomic components, pairwise fusions, and bridge synergies

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

Run the system with default settings (10 iterations):

```bash
python -m src.main
```

Customize the run with command-line arguments:

```bash
python -m src.main --prompt "Your initial prompt" --iterations 20 --output output_directory
```

Load a previous state and continue reasoning:

```bash
python -m src.main --load output/state_20250320123456.json --iterations 10
```

Run in quiet mode (less verbose output):

```bash
python -m src.main --quiet
```

## Output

The system generates several output files:
- Knowledge graph visualization (`graph_*.png`)
- System state JSON (`state_*.json`)
- Summary report (`summary_*.md`)

## Implementation Details

The system consists of several key components:

1. **KnowledgeGraph**: Manages the graph structure with nodes, edges, and centrality metrics
2. **SemanticHub**: Handles cross-modal representations and similarity calculations
3. **CompositionalReasoning**: Performs higher-order reasoning over the graph
4. **ReasoningSystem**: Orchestrates the reasoning process and manages state

The reasoning loop follows these steps:
1. Generate reasoning tokens based on the current task
2. Extract a local graph from the reasoning
3. Merge the local graph into the global graph
4. Update semantic representations
5. Update node centrality metrics
6. Periodically update compositional reasoning
7. Generate the next question

## License

MIT License 
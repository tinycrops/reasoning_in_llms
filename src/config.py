"""
Configuration for the Agentic Reasoning System based on the TLA+ spec
"""

# System constants
MAX_ITERATIONS = 100
MAX_NODES = 1000
MAX_EDGES = 5000
INITIAL_PROMPT = "Explore applications of nanotechnology in sustainable energy generation."

# Model configuration
LLM_MODELS = {
    "gemini-pro": {
        "provider": "gemini",
        "model": "gemini-2.0-flash",
        "max_tokens": 8192,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_tokens": 4096,
    }
}

# Graph configuration
NODE_TYPES = [
    "Concept", 
    "Question", 
    "Hypothesis", 
    "Evidence", 
    "Method", 
    "Application",
    "Challenge",
    "Principle"
]

EDGE_TYPES = [
    "relates_to",
    "causes",
    "supports",
    "contradicts",
    "exemplifies",
    "generalizes",
    "precedes",
    "enables",
    "inhibits",
    "similar_to",
    "part_of"
]

# Input modalities (simplified for initial implementation)
MODALITIES = ["Text"]  # Could expand to ["Text", "Image", "Audio"] in future

# Vector embedding dimensions (for semantic hub)
EMBEDDING_DIM = 768

# Reasoning system parameters
REASONING_TEMPERATURE = 0.7
GRAPH_EXTRACTION_TEMPERATURE = 0.2
COMPOSITIONAL_TEMPERATURE = 0.9
MODEL_SWITCH_INTERVAL = 10  # Switch models every N iterations
COMPOSITIONAL_UPDATE_INTERVAL = 5  # Update compositional state every N iterations

# Prompts
REASONING_PROMPT_TEMPLATE = """
You are part of an agentic reasoning system that builds knowledge iteratively. 
Current task: {current_task}
Current iteration: {iteration} of {max_iterations}

Your role is to think deeply about this topic and generate insightful thoughts.
Be creative, analytical, and consider multiple perspectives.

Current graph summary:
{graph_summary}

Think step-by-step and generate reasoning:
"""

GRAPH_EXTRACTION_PROMPT = """
Based on the reasoning below, extract a set of concepts (nodes) and relationships (edges) that would effectively represent this knowledge in a graph structure.

Reasoning:
{reasoning_text}

For each node, provide:
1. A short title/concept name
2. The type of node (from: {node_types})
3. A clear description of the concept

For each edge, provide:
1. The source node title
2. The target node title
3. The type of relationship (from: {edge_types})
4. A description of how they're related

Format your response as a JSON object with "nodes" and "edges" arrays.
"""

COMPOSITIONAL_REASONING_PROMPT = """
You are analyzing a knowledge graph to perform compositional reasoning.

Here are some key concepts from the graph:
{key_nodes}

Your task is to:
1. Identify atomic components (fundamental ideas)
2. Generate pairwise fusions (how two concepts can combine to create new insights)
3. Identify bridge synergies (how multiple concepts work together)
4. Create a final synthesis that integrates these insights

Think deeply about how these concepts might interact in novel ways.
""" 
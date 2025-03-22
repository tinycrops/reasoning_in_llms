"""
Main Reasoning System Implementation
"""
import json
import os
import time
from datetime import datetime
from src.models.knowledge_graph import KnowledgeGraph
from src.models.semantic_hub import SemanticHub
from src.models.compositional_reasoning import CompositionalReasoning
from src.utils.api_handlers import generate_with_llm, extract_json_from_response
from src.config import (
    MAX_ITERATIONS, MAX_NODES, MAX_EDGES, INITIAL_PROMPT,
    LLM_MODELS, NODE_TYPES, EDGE_TYPES, MODALITIES,
    REASONING_TEMPERATURE, GRAPH_EXTRACTION_TEMPERATURE, COMPOSITIONAL_TEMPERATURE,
    MODEL_SWITCH_INTERVAL, COMPOSITIONAL_UPDATE_INTERVAL,
    REASONING_PROMPT_TEMPLATE, GRAPH_EXTRACTION_PROMPT
)

class ReasoningSystem:
    """
    Main implementation of the Agentic Reasoning System
    """
    
    def __init__(self, initial_prompt=None):
        """
        Initialize the reasoning system
        
        Args:
            initial_prompt: The initial prompt to seed the reasoning
        """
        self.initial_prompt = initial_prompt or INITIAL_PROMPT
        self.graph = KnowledgeGraph(self.initial_prompt)
        self.semantic_hub = SemanticHub(MODALITIES)
        self.compositional_reasoning = CompositionalReasoning()
        self.active_models = list(LLM_MODELS.keys())[0:1]  # Start with the first model
        self.iterations = 0
        self.reasoning = {
            "current_task": self.initial_prompt,
            "thinking_tokens": "",
            "extracted_graph": {"nodes": set(), "edges": set()}
        }
        
        # Add initial node representation
        self.semantic_hub.add_node_representation(0, self.initial_prompt)
    
    def run_iteration(self, verbose=True):
        """
        Run a single reasoning iteration
        
        Args:
            verbose: Whether to print verbose output
        """
        if self.iterations >= MAX_ITERATIONS:
            if verbose:
                print("Maximum iterations reached.")
            return False
        
        # Generate reasoning tokens
        if not self._generate_reasoning_tokens(verbose):
            return False
        
        # Extract local graph
        if not self._extract_local_graph(verbose):
            return False
        
        # Merge graphs
        if not self._merge_graphs(verbose):
            return False
        
        # Update semantic hub
        if not self._update_semantic_hub(verbose):
            return False
        
        # Update node centrality
        if not self._update_node_centrality(verbose):
            return False
        
        # Update compositional reasoning (every N iterations)
        if self.iterations % COMPOSITIONAL_UPDATE_INTERVAL == 0:
            if not self._update_compositional_reasoning(verbose):
                return False
        
        # Generate next question
        if not self._generate_next_question(verbose):
            return False
        
        # Switch model (every N iterations)
        if self.iterations % MODEL_SWITCH_INTERVAL == 0:
            if not self._switch_model(verbose):
                return False
        
        # Increment iteration counter
        self.iterations += 1
        
        return True
    
    def run_iterations(self, num_iterations=MAX_ITERATIONS, verbose=True):
        """
        Run multiple iterations
        
        Args:
            num_iterations: Number of iterations to run
            verbose: Whether to print verbose output
        """
        start_time = time.time()
        
        for i in range(num_iterations):
            if verbose:
                print(f"\n=== Iteration {self.iterations + 1}/{MAX_ITERATIONS} ===")
            
            if not self.run_iteration(verbose):
                break
                
            if verbose:
                print(f"Completed iteration {self.iterations}/{MAX_ITERATIONS}")
                
        end_time = time.time()
        if verbose:
            print(f"\nCompleted {self.iterations} iterations in {end_time - start_time:.2f} seconds.")
    
    def _get_current_model_info(self):
        """
        Get the current active model info
        """
        model_name = self.active_models[0]
        return LLM_MODELS[model_name]
    
    def _generate_reasoning_tokens(self, verbose=True):
        """
        Generate reasoning tokens based on current task
        """
        if verbose:
            print("Generating reasoning tokens...")
        
        # Get model info
        model_info = self._get_current_model_info()
        
        # Prepare prompt
        prompt = REASONING_PROMPT_TEMPLATE.format(
            current_task=self.reasoning["current_task"],
            iteration=self.iterations,
            max_iterations=MAX_ITERATIONS,
            graph_summary=self.graph.get_graph_summary()
        )
        
        # Generate reasoning
        response = generate_with_llm(
            prompt=prompt,
            provider=model_info["provider"],
            model=model_info["model"],
            temperature=REASONING_TEMPERATURE,
            max_tokens=model_info["max_tokens"] // 2  # Use half the max tokens
        )
        
        if not response:
            if verbose:
                print("Failed to generate reasoning tokens.")
            return False
        
        # Store the reasoning
        self.reasoning["thinking_tokens"] = response
        
        if verbose:
            print(f"Generated {len(response.split())} tokens of reasoning.")
        
        return True
    
    def _extract_local_graph(self, verbose=True):
        """
        Extract a local graph from the reasoning
        """
        if verbose:
            print("Extracting local graph...")
        
        # Get model info - use gpt-4-mini for structured extraction
        model_info = LLM_MODELS["gpt-4o-mini"]
        
        # Prepare prompt
        prompt = GRAPH_EXTRACTION_PROMPT.format(
            reasoning_text=self.reasoning["thinking_tokens"],
            node_types=", ".join(NODE_TYPES),
            edge_types=", ".join(EDGE_TYPES)
        )
        
        # Generate extraction
        response = generate_with_llm(
            prompt=prompt,
            provider=model_info["provider"],
            model=model_info["model"],
            temperature=GRAPH_EXTRACTION_TEMPERATURE,
            max_tokens=model_info["max_tokens"] // 2
        )
        
        if not response:
            if verbose:
                print("Failed to extract local graph.")
            return False
        
        # Parse the response to extract nodes and edges
        extracted_data = extract_json_from_response(response)
        
        if not extracted_data or "nodes" not in extracted_data or "edges" not in extracted_data:
            if verbose:
                print("Failed to parse extracted graph data.")
            return False
        
        # Store the extracted graph data
        self.reasoning["extracted_graph"] = {
            "nodes": set(),
            "edges": set()
        }
        
        # Process the extract nodes and edges
        # We don't convert to graph nodes yet, as we need to merge with the main graph first
        node_map = {}  # Map from node titles to their data
        
        for node in extracted_data["nodes"]:
            if not isinstance(node, dict):
                continue
                
            # Extract required fields with defaults
            title = node.get("title", "")
            
            # Skip if no title
            if not title:
                continue
                
            node_type = node.get("type", "Concept")
            # Validate node type
            if node_type not in NODE_TYPES:
                node_type = "Concept"
                
            content = node.get("description", title)
            
            # Store the node data
            node_map[title] = {
                "type": node_type,
                "content": content
            }
            
            # Add to extracted graph nodes
            self.reasoning["extracted_graph"]["nodes"].add(title)
        
        # Process edges
        for edge in extracted_data["edges"]:
            if not isinstance(edge, dict):
                continue
                
            # Extract required fields
            source = edge.get("source", "")
            target = edge.get("target", "")
            
            # Skip if source or target is missing or not in node_map
            if not source or not target or source not in node_map or target not in node_map:
                continue
                
            edge_type = edge.get("type", "relates_to")
            
            # Validate edge type
            if edge_type not in EDGE_TYPES:
                edge_type = "relates_to"
                
            # Add to extracted graph edges
            self.reasoning["extracted_graph"]["edges"].add((source, target, edge_type))
        
        if verbose:
            print(f"Extracted {len(self.reasoning['extracted_graph']['nodes'])} nodes and "
                  f"{len(self.reasoning['extracted_graph']['edges'])} edges.")
            
        # Store the node_map for use in merge_graphs
        self.reasoning["node_map"] = node_map
        
        return True
    
    def _merge_graphs(self, verbose=True):
        """
        Merge the extracted graph into the main graph
        """
        if verbose:
            print("Merging graphs...")
        
        if not self.reasoning["extracted_graph"]["nodes"]:
            if verbose:
                print("No nodes to merge.")
            return True
        
        # Track new nodes and their IDs
        new_node_ids = {}
        
        # Add nodes to the main graph
        for node_title in self.reasoning["extracted_graph"]["nodes"]:
            node_data = self.reasoning["node_map"][node_title]
            
            # Check if we've reached the maximum number of nodes
            if len(self.graph.graph.nodes) >= MAX_NODES:
                if verbose:
                    print(f"Maximum nodes ({MAX_NODES}) reached. Skipping remaining nodes.")
                break
                
            # Add the node to the graph
            node_id = self.graph.add_node(node_data["type"], node_data["content"])
            new_node_ids[node_title] = node_id
        
        # Add edges to the main graph
        for source_title, target_title, edge_type in self.reasoning["extracted_graph"]["edges"]:
            # Skip if either source or target is not in the new_node_ids
            if source_title not in new_node_ids or target_title not in new_node_ids:
                continue
                
            # Check if we've reached the maximum number of edges
            if len(self.graph.graph.edges) >= MAX_EDGES:
                if verbose:
                    print(f"Maximum edges ({MAX_EDGES}) reached. Skipping remaining edges.")
                break
                
            # Add the edge to the graph
            self.graph.add_edge(
                new_node_ids[source_title],
                new_node_ids[target_title],
                edge_type
            )
        
        if verbose:
            print(f"Merged {len(new_node_ids)} nodes and {len(self.reasoning['extracted_graph']['edges'])} edges.")
        
        return True
    
    def _update_semantic_hub(self, verbose=True):
        """
        Update the semantic hub with new node representations
        """
        if verbose:
            print("Updating semantic hub...")
        
        # Find nodes that don't have representations yet
        all_node_ids = set(self.graph.graph.nodes())
        nodes_with_embeddings = set()
        
        for modality in self.semantic_hub.modalities:
            nodes_with_embeddings.update(self.semantic_hub.representations[modality].keys())
        
        new_node_ids = all_node_ids - nodes_with_embeddings
        
        # Add representations for new nodes
        for node_id in new_node_ids:
            node = self.graph.get_node(node_id)
            if node:
                self.semantic_hub.add_node_representation(node_id, node["content"])
        
        if verbose:
            print(f"Added representations for {len(new_node_ids)} new nodes.")
        
        return True
    
    def _update_node_centrality(self, verbose=True):
        """
        Update node centrality metrics
        """
        if verbose:
            print("Updating node centrality...")
        
        self.graph.update_centrality_metrics()
        
        if verbose:
            print("Node centrality updated.")
        
        return True
    
    def _update_compositional_reasoning(self, verbose=True):
        """
        Update compositional reasoning
        """
        if verbose:
            print("Updating compositional reasoning...")
        
        # Get model info - use gemini for creative synthesis
        model_info = LLM_MODELS["gemini-pro"]
        
        # Update compositional reasoning
        success = self.compositional_reasoning.update_from_graph(
            self.graph,
            top_n=5,
            provider=model_info["provider"],
            model=model_info["model"],
            temperature=COMPOSITIONAL_TEMPERATURE
        )
        
        if not success:
            if verbose:
                print("Failed to update compositional reasoning.")
            return False
        
        if verbose:
            print("Compositional reasoning updated.")
            print(self.compositional_reasoning.get_summary())
        
        return True
    
    def _generate_next_question(self, verbose=True):
        """
        Generate the next question based on the graph
        """
        if verbose:
            print("Generating next question...")
        
        # Get model info
        model_info = self._get_current_model_info()
        
        # Get top nodes by centrality
        top_node_ids = self.graph.get_top_nodes_by_centrality(3)
        
        # Prepare node content for prompt
        node_contents = []
        for node_id in top_node_ids:
            node = self.graph.get_node(node_id)
            if node:
                node_contents.append(f"{node['content']} (Type: {node['type']})")
        
        # Generate prompt for next question
        prompt = f"""
Based on the following key concepts, generate a new question or task for the reasoning system to explore next.
The question should be focused, specific, and build upon the concepts below.

Key concepts:
{chr(10).join(['- ' + content for content in node_contents])}

Current iteration: {self.iterations + 1} of {MAX_ITERATIONS}

Generate a concise, focused question that will help expand the knowledge graph in an interesting direction:
"""
        
        # Generate next question
        response = generate_with_llm(
            prompt=prompt,
            provider=model_info["provider"],
            model=model_info["model"],
            temperature=REASONING_TEMPERATURE,
            max_tokens=100  # Short response for the next question
        )
        
        if not response:
            if verbose:
                print("Failed to generate next question.")
            return False
        
        # Store the next question
        self.reasoning["current_task"] = response.strip()
        
        if verbose:
            print(f"Next question: {self.reasoning['current_task']}")
        
        return True
    
    def _switch_model(self, verbose=True):
        """
        Switch the active model
        """
        if verbose:
            print("Switching model...")
        
        # Get current model
        current_model = self.active_models[0]
        
        # Get other available models
        other_models = [model for model in LLM_MODELS.keys() if model != current_model]
        
        if not other_models:
            if verbose:
                print("No other models available.")
            return True
        
        # Switch to the first other model
        self.active_models = [other_models[0]]
        
        if verbose:
            print(f"Switched to model: {self.active_models[0]}")
        
        return True
    
    def save_state(self, output_dir="output"):
        """
        Save the current state of the system
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Save graph visualization
        graph_viz_path = os.path.join(output_dir, f"graph_{timestamp}.png")
        self.graph.visualize(graph_viz_path)
        
        # Save system state
        state = {
            "iterations": self.iterations,
            "initial_prompt": self.initial_prompt,
            "current_task": self.reasoning["current_task"],
            "active_models": self.active_models,
            "graph": self.graph.to_dict(),
            "semantic_hub": self.semantic_hub.to_dict(),
            "compositional_reasoning": self.compositional_reasoning.to_dict()
        }
        
        state_path = os.path.join(output_dir, f"state_{timestamp}.json")
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save a summary report
        summary = []
        summary.append(f"# Agentic Reasoning System - Summary Report ({timestamp})")
        summary.append(f"\nInitial Prompt: {self.initial_prompt}")
        summary.append(f"\nIterations Completed: {self.iterations}")
        summary.append(f"\nFinal Task: {self.reasoning['current_task']}")
        summary.append(f"\n## Graph Summary")
        summary.append(self.graph.get_graph_summary(max_nodes=10))
        summary.append(f"\n## Compositional Reasoning Summary")
        summary.append(self.compositional_reasoning.get_summary())
        summary.append(f"\n## Statistics")
        summary.append(f"- Total Nodes: {len(self.graph.graph.nodes)}")
        summary.append(f"- Total Edges: {len(self.graph.graph.edges)}")
        
        try:
            summary.append(f"- Average Shortest Path Length: {self.graph.get_average_shortest_path_length():.2f}")
        except:
            summary.append(f"- Average Shortest Path Length: N/A")
            
        summary.append(f"- Bridge Nodes: {len(self.graph.get_bridge_nodes())}")
        summary.append("\n## Node Degree Distribution")
        
        degree_dist = self.graph.get_node_degree_distribution()
        for degree, count in sorted(degree_dist.items()):
            summary.append(f"- Degree {degree}: {count} nodes")
        
        summary_path = os.path.join(output_dir, f"summary_{timestamp}.md")
        with open(summary_path, "w") as f:
            f.write("\n".join(summary))
        
        return {
            "graph_viz": graph_viz_path,
            "state": state_path,
            "summary": summary_path
        }
    
    @classmethod
    def load_state(cls, state_file):
        """
        Load the system state from a file
        """
        with open(state_file, "r") as f:
            state = json.load(f)
        
        # Create a new instance
        system = cls(state["initial_prompt"])
        
        # Restore state
        system.iterations = state["iterations"]
        system.reasoning["current_task"] = state["current_task"]
        system.active_models = state["active_models"]
        
        # Restore graph
        system.graph = KnowledgeGraph.from_dict(state["graph"])
        
        # Restore semantic hub
        system.semantic_hub = SemanticHub.from_dict(state["semantic_hub"])
        
        # Restore compositional reasoning
        system.compositional_reasoning = CompositionalReasoning.from_dict(state["compositional_reasoning"])
        
        return system 
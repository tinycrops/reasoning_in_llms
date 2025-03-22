"""
Compositional Reasoning Module
"""
import json
from src.utils.api_handlers import generate_with_llm, extract_json_from_response
from src.config import COMPOSITIONAL_REASONING_PROMPT

class CompositionalReasoning:
    """
    Implementation of the Compositional Reasoning process
    """
    
    def __init__(self):
        """
        Initialize the compositional reasoning state
        """
        self.atomic_components = set()
        self.pairwise_fusions = []
        self.bridge_synergies = []
        self.final_synthesis = ""
    
    def update_from_graph(self, graph, top_n=5, provider="gemini", model="gemini-pro", temperature=0.9):
        """
        Update compositional reasoning based on the current state of the graph
        
        Args:
            graph: KnowledgeGraph instance
            top_n: Number of top nodes to consider
            provider: LLM provider (gemini or openai)
            model: Model name
            temperature: Temperature for generation
        """
        # Get top nodes by centrality
        top_node_ids = graph.get_top_nodes_by_centrality(top_n, "eigenvector")
        
        # Prepare key nodes info
        key_nodes = []
        for node_id in top_node_ids:
            node = graph.get_node(node_id)
            if node:
                key_nodes.append({
                    "id": node_id,
                    "type": node["type"],
                    "content": node["content"],
                    "neighbors": [
                        {
                            "id": neighbor_id,
                            "type": graph.get_node(neighbor_id)["type"],
                            "content": graph.get_node(neighbor_id)["content"],
                            "relation": edge_type
                        }
                        for neighbor_id, edge_type in graph.get_neighbors(node_id)[:3]  # Limit to top 3 neighbors
                    ]
                })
        
        # Format nodes for prompt
        formatted_nodes = "\n".join([
            f"Node {i+1}: {node['content']} (Type: {node['type']})"
            for i, node in enumerate(key_nodes)
        ])
        
        # Generate compositional reasoning
        prompt = COMPOSITIONAL_REASONING_PROMPT.format(
            key_nodes=formatted_nodes
        )
        
        response = generate_with_llm(
            prompt=prompt,
            provider=provider,
            model=model,
            temperature=temperature
        )
        
        if not response:
            return False
        
        # Process the response to extract components
        self._process_compositional_response(response, key_nodes)
        
        return True
    
    def _process_compositional_response(self, response, key_nodes):
        """
        Process the LLM's response to extract compositional reasoning elements
        """
        # Extract atomic components
        atomic_section = self._extract_section(response, "atomic components", "pairwise fusions")
        if atomic_section:
            self.atomic_components = set([
                node["id"] for node in key_nodes if any(
                    keyword.lower() in node["content"].lower() 
                    for keyword in self._extract_keywords(atomic_section)
                )
            ])
        
        # Extract pairwise fusions
        fusion_section = self._extract_section(response, "pairwise fusions", "bridge synergies")
        if fusion_section:
            self.pairwise_fusions = self._extract_pairwise_fusions(fusion_section, key_nodes)
        
        # Extract bridge synergies
        synergy_section = self._extract_section(response, "bridge synergies", "final synthesis")
        if synergy_section:
            self.bridge_synergies = self._extract_bridge_synergies(synergy_section, key_nodes)
        
        # Extract final synthesis
        synthesis_section = self._extract_section(response, "final synthesis", None)
        if synthesis_section:
            self.final_synthesis = synthesis_section.strip()
    
    def _extract_section(self, text, start_marker, end_marker=None):
        """
        Extract a section from the text
        """
        start_idx = text.lower().find(start_marker.lower())
        
        if start_idx == -1:
            return ""
        
        # Find the end of the heading line
        start_idx = text.find("\n", start_idx)
        if start_idx == -1:
            start_idx = len(text)
        else:
            start_idx += 1
        
        if end_marker:
            end_idx = text.lower().find(end_marker.lower(), start_idx)
            if end_idx == -1:
                return text[start_idx:].strip()
            return text[start_idx:end_idx].strip()
        else:
            return text[start_idx:].strip()
    
    def _extract_keywords(self, text):
        """
        Extract key concept words from a section
        """
        # This is a simplistic approach - in practice, you'd want more sophisticated NLP
        words = []
        lines = text.split("\n")
        for line in lines:
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                item = line.strip()[2:].strip()
                words.append(item)
        
        # If no bullet points found, try to extract phrases
        if not words:
            import re
            words = re.findall(r'"([^"]*)"', text)
            
        # If still no phrases, take first words of sentences
        if not words:
            sentences = text.split(". ")
            words = [s.split()[0] for s in sentences if s.strip()]
        
        return words
    
    def _extract_pairwise_fusions(self, text, key_nodes):
        """
        Extract pairwise fusions from text
        """
        fusions = []
        lines = text.split("\n")
        current_fusion = {"node1": None, "node2": None, "fusion": ""}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for lines with node references
            node_refs = []
            for i, node in enumerate(key_nodes):
                node_num = i + 1
                if f"Node {node_num}" in line or f"node {node_num}" in line:
                    node_refs.append(node["id"])
            
            if len(node_refs) >= 2:
                # This line identifies a fusion between nodes
                current_fusion = {
                    "node1": node_refs[0],
                    "node2": node_refs[1],
                    "fusion": line
                }
                fusions.append(current_fusion)
            elif current_fusion["node1"] is not None:
                # Add to the description of the current fusion
                current_fusion["fusion"] += " " + line
        
        return fusions
    
    def _extract_bridge_synergies(self, text, key_nodes):
        """
        Extract bridge synergies from text
        """
        synergies = []
        lines = text.split("\n")
        current_synergy = {"nodes": set(), "synergy": ""}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for lines with node references
            node_refs = []
            for i, node in enumerate(key_nodes):
                node_num = i + 1
                if f"Node {node_num}" in line or f"node {node_num}" in line:
                    node_refs.append(node["id"])
            
            if len(node_refs) >= 2:
                # This line identifies a synergy among nodes
                current_synergy = {
                    "nodes": set(node_refs),
                    "synergy": line
                }
                synergies.append(current_synergy)
            elif current_synergy["nodes"]:
                # Add to the description of the current synergy
                current_synergy["synergy"] += " " + line
        
        return synergies
    
    def get_summary(self):
        """
        Get a summary of the compositional reasoning state
        """
        summary = []
        
        if self.atomic_components:
            summary.append(f"Atomic Components: {len(self.atomic_components)} concepts identified")
        
        if self.pairwise_fusions:
            summary.append(f"Pairwise Fusions: {len(self.pairwise_fusions)} novel combinations created")
        
        if self.bridge_synergies:
            summary.append(f"Bridge Synergies: {len(self.bridge_synergies)} multi-concept synergies discovered")
        
        if self.final_synthesis:
            summary.append(f"Final Synthesis: {self.final_synthesis[:100]}...")
        
        return "\n".join(summary) if summary else "No compositional reasoning performed yet."
    
    def to_dict(self):
        """
        Convert compositional reasoning to a dictionary
        """
        return {
            "atomic_components": list(self.atomic_components),
            "pairwise_fusions": [
                {
                    "node1": fusion["node1"],
                    "node2": fusion["node2"],
                    "fusion": fusion["fusion"]
                }
                for fusion in self.pairwise_fusions
            ],
            "bridge_synergies": [
                {
                    "nodes": list(synergy["nodes"]),
                    "synergy": synergy["synergy"]
                }
                for synergy in self.bridge_synergies
            ],
            "final_synthesis": self.final_synthesis
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create compositional reasoning from a dictionary
        """
        comp = cls()
        
        comp.atomic_components = set(data.get("atomic_components", []))
        
        comp.pairwise_fusions = [
            {
                "node1": fusion["node1"],
                "node2": fusion["node2"],
                "fusion": fusion["fusion"]
            }
            for fusion in data.get("pairwise_fusions", [])
        ]
        
        comp.bridge_synergies = [
            {
                "nodes": set(synergy["nodes"]),
                "synergy": synergy["synergy"]
            }
            for synergy in data.get("bridge_synergies", [])
        ]
        
        comp.final_synthesis = data.get("final_synthesis", "")
        
        return comp 
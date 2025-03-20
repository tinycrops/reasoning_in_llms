"""
Knowledge Graph Implementation
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class KnowledgeGraph:
    """
    Implementation of the knowledge graph as described in the TLA+ spec
    """
    
    def __init__(self, initial_content=None):
        """
        Initialize an empty knowledge graph
        """
        self.graph = nx.DiGraph()
        self.next_node_id = 0
        
        # Add the initial node if provided
        if initial_content:
            self.add_node("Concept", initial_content)
    
    def add_node(self, node_type, content):
        """
        Add a node to the graph
        """
        node_id = self.next_node_id
        self.graph.add_node(
            node_id,
            id=node_id,
            type=node_type,
            content=content,
            centrality={
                "betweenness": 0.0,
                "eigenvector": 0.0,
                "closeness": 0.0
            }
        )
        self.next_node_id += 1
        return node_id
    
    def add_edge(self, source_id, target_id, edge_type, weight=1.0):
        """
        Add an edge between two nodes
        """
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return False
        
        self.graph.add_edge(
            source_id,
            target_id,
            type=edge_type,
            weight=weight
        )
        return True
    
    def get_node(self, node_id):
        """
        Get node by ID
        """
        if node_id in self.graph.nodes:
            return self.graph.nodes[node_id]
        return None
    
    def get_neighbors(self, node_id):
        """
        Get all neighbors of a node
        """
        if node_id not in self.graph.nodes:
            return []
        
        neighbors = []
        for neighbor in self.graph.successors(node_id):
            neighbors.append((neighbor, self.graph.edges[node_id, neighbor]['type']))
        for neighbor in self.graph.predecessors(node_id):
            if neighbor != node_id:  # Avoid self-loops
                neighbors.append((neighbor, self.graph.edges[neighbor, node_id]['type']))
        
        return neighbors
    
    def update_centrality_metrics(self):
        """
        Update centrality metrics for all nodes
        """
        if len(self.graph.nodes) <= 1:
            return
        
        # Calculate centrality metrics
        try:
            betweenness = nx.betweenness_centrality(self.graph)
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)
            closeness = nx.closeness_centrality(self.graph)
            
            # Update node attributes
            for node_id in self.graph.nodes:
                self.graph.nodes[node_id]['centrality'] = {
                    "betweenness": betweenness.get(node_id, 0.0),
                    "eigenvector": eigenvector.get(node_id, 0.0),
                    "closeness": closeness.get(node_id, 0.0)
                }
        except:
            # Fallback for cases where the algorithms fail
            # (e.g., disconnected graph for eigenvector centrality)
            for node_id in self.graph.nodes:
                degree = self.graph.degree(node_id)
                num_nodes = len(self.graph.nodes)
                self.graph.nodes[node_id]['centrality'] = {
                    "betweenness": degree / max(num_nodes, 1),
                    "eigenvector": degree / max(num_nodes, 1),
                    "closeness": degree / max(num_nodes, 1)
                }
    
    def get_top_nodes_by_centrality(self, n=5, metric="eigenvector"):
        """
        Get top N nodes by a specific centrality metric
        """
        nodes = list(self.graph.nodes)
        nodes.sort(key=lambda x: self.graph.nodes[x]['centrality'][metric], reverse=True)
        return nodes[:min(n, len(nodes))]
    
    def get_community_structure(self):
        """
        Detect communities in the graph
        """
        if len(self.graph.nodes) <= 1:
            return {0: [n for n in self.graph.nodes]}
            
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Use Louvain method for community detection
        try:
            from community import best_partition
            return best_partition(undirected)
        except:
            # Fallback to connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(undirected)):
                for node in component:
                    communities[node] = i
            return communities
    
    def get_bridge_nodes(self):
        """
        Identify bridge nodes (nodes connecting different communities)
        """
        communities = self.get_community_structure()
        
        bridge_nodes = []
        for node in self.graph.nodes:
            node_community = communities.get(node)
            neighbor_communities = set()
            
            for neighbor in self.graph.neighbors(node):
                neighbor_community = communities.get(neighbor)
                if neighbor_community != node_community:
                    neighbor_communities.add(neighbor_community)
            
            if len(neighbor_communities) > 0:
                bridge_nodes.append((node, len(neighbor_communities)))
        
        # Sort by number of different communities connected
        bridge_nodes.sort(key=lambda x: x[1], reverse=True)
        return bridge_nodes
    
    def get_average_shortest_path_length(self):
        """
        Calculate average shortest path length
        """
        if len(self.graph.nodes) <= 1:
            return 0
        
        # Convert to undirected for shortest path calculation
        undirected = self.graph.to_undirected()
        
        # Check if graph is connected
        if not nx.is_connected(undirected):
            # Calculate for each connected component
            components = list(nx.connected_components(undirected))
            avg_path_lengths = []
            
            for component in components:
                if len(component) > 1:
                    subgraph = undirected.subgraph(component)
                    avg_path_lengths.append(nx.average_shortest_path_length(subgraph))
            
            return np.mean(avg_path_lengths) if avg_path_lengths else 0
        else:
            return nx.average_shortest_path_length(undirected)
    
    def get_node_degree_distribution(self):
        """
        Get the degree distribution of the graph
        """
        degrees = [d for _, d in self.graph.degree()]
        return Counter(degrees)
    
    def get_graph_summary(self, max_nodes=5):
        """
        Get a textual summary of the graph
        """
        if len(self.graph.nodes) == 0:
            return "Empty graph."
        
        summary = []
        summary.append(f"Graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
        
        # Add top central nodes
        if len(self.graph.nodes) > 1:
            top_nodes = self.get_top_nodes_by_centrality(max_nodes)
            summary.append("Top central concepts:")
            for node_id in top_nodes:
                node = self.graph.nodes[node_id]
                summary.append(f"- {node['content']} (type: {node['type']})")
        
        # Add information about communities
        if len(self.graph.nodes) > 2:
            communities = self.get_community_structure()
            num_communities = len(set(communities.values()))
            summary.append(f"Number of distinct concept clusters: {num_communities}")
        
        # Add information about bridge nodes
        if len(self.graph.nodes) > 2:
            bridge_nodes = self.get_bridge_nodes()
            if bridge_nodes:
                summary.append("Key bridging concepts:")
                for node_id, _ in bridge_nodes[:min(3, len(bridge_nodes))]:
                    node = self.graph.nodes[node_id]
                    summary.append(f"- {node['content']}")
        
        return "\n".join(summary)
    
    def visualize(self, output_file="knowledge_graph.png"):
        """
        Visualize the graph
        """
        plt.figure(figsize=(12, 10))
        
        # Get position layout
        pos = nx.spring_layout(self.graph)
        
        # Get node colors based on type
        node_types = [self.graph.nodes[n]['type'] for n in self.graph.nodes]
        unique_types = list(set(node_types))
        color_map = {t: plt.cm.tab10(i/len(unique_types)) for i, t in enumerate(unique_types)}
        node_colors = [color_map[t] for t in node_types]
        
        # Get node sizes based on eigenvector centrality
        node_sizes = [1000 * (0.1 + self.graph.nodes[n]['centrality']['eigenvector']) for n in self.graph.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        # Draw edges
        edge_types = [self.graph.edges[e]['type'] for e in self.graph.edges]
        unique_edge_types = list(set(edge_types))
        edge_color_map = {t: plt.cm.Set2(i/len(unique_edge_types)) for i, t in enumerate(unique_edge_types)}
        edge_colors = [edge_color_map[t] for t in edge_types]
        
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=1.5, alpha=0.7, arrows=True, arrowsize=15)
        
        # Add labels
        labels = {n: self.graph.nodes[n].get('content', '')[:15] + '...' for n in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)
        
        # Add legend for node types
        node_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[t], 
                                 markersize=10, label=f"Node: {t}") for t in unique_types]
        edge_patches = [plt.Line2D([0], [0], color=edge_color_map[t], lw=2, label=f"Edge: {t}") 
                       for t in unique_edge_types]
        
        plt.legend(handles=node_patches + edge_patches, loc='upper right', fontsize=8)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def to_dict(self):
        """
        Convert the graph to a dictionary
        """
        return {
            "nodes": [
                {
                    "id": n,
                    "type": self.graph.nodes[n]["type"],
                    "content": self.graph.nodes[n]["content"],
                    "centrality": self.graph.nodes[n]["centrality"]
                }
                for n in self.graph.nodes
            ],
            "edges": [
                {
                    "source": e[0],
                    "target": e[1],
                    "type": self.graph.edges[e]["type"],
                    "weight": self.graph.edges[e].get("weight", 1.0)
                }
                for e in self.graph.edges
            ]
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a graph from a dictionary
        """
        graph = cls()
        
        # Clear the graph and reset node ID counter
        graph.graph = nx.DiGraph()
        
        # Add nodes
        max_id = -1
        for node_data in data["nodes"]:
            node_id = node_data["id"]
            graph.graph.add_node(
                node_id,
                id=node_id,
                type=node_data["type"],
                content=node_data["content"],
                centrality=node_data["centrality"]
            )
            max_id = max(max_id, node_id)
        
        # Set next node ID
        graph.next_node_id = max_id + 1
        
        # Add edges
        for edge_data in data["edges"]:
            graph.graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                type=edge_data["type"],
                weight=edge_data.get("weight", 1.0)
            )
        
        return graph 
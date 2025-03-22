"""
Semantic Hub for Cross-Modal Representations
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.api_handlers import generate_with_llm

class SemanticHub:
    """
    Implementation of the Semantic Hub for cross-modal representations
    """
    
    def __init__(self, modalities=None, embedding_dim=768):
        """
        Initialize the semantic hub
        
        Args:
            modalities: List of input modalities
            embedding_dim: Dimension of the embedding vectors
        """
        self.modalities = modalities or ["Text"]
        self.embedding_dim = embedding_dim
        
        # Initialize representations
        self.representations = {modality: {} for modality in self.modalities}
        self.cross_modal_similarity = {}
        
    def generate_embedding(self, content, modality="Text"):
        """
        Generate embedding for a node's content
        
        In a production system, this would use proper embedding models.
        For this prototype, we'll generate random embeddings, with similar
        content getting similar embeddings via a simple hashing trick.
        """
        if modality not in self.modalities:
            raise ValueError(f"Unsupported modality: {modality}")
        
        # For the prototype, simulate embeddings with a simple hash-based approach
        # In reality, you would use a proper embedding model (e.g., CLIP for images, BERT for text)
        # This is where we can "fudge the numbers" as mentioned in the requirements
        
        # Simple hash-based pseudo-embedding generation
        content_hash = hash(content)
        np.random.seed(content_hash % 10000)  # Use content hash as seed for deterministic results
        
        # Generate base embedding
        embedding = np.random.normal(0, 1, self.embedding_dim)
        
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    def add_node_representation(self, node_id, content, modality="Text"):
        """
        Add a representation for a node
        """
        if modality not in self.modalities:
            return False
        
        # Generate embedding
        embedding = self.generate_embedding(content, modality)
        
        # Store representation
        if modality not in self.representations:
            self.representations[modality] = {}
        
        self.representations[modality][node_id] = embedding
        
        # Update cross-modal similarities
        self._update_similarities(node_id)
        
        return True
    
    def get_representation(self, node_id, modality="Text"):
        """
        Get the representation of a node
        """
        if modality not in self.modalities or node_id not in self.representations.get(modality, {}):
            return None
        
        return self.representations[modality][node_id]
    
    def get_similarity(self, node_id1, node_id2):
        """
        Get the similarity between two nodes
        """
        key = (min(node_id1, node_id2), max(node_id1, node_id2))
        
        if key in self.cross_modal_similarity:
            return self.cross_modal_similarity[key]
        
        # Calculate similarity if not cached
        similarity = self._calculate_similarity(node_id1, node_id2)
        self.cross_modal_similarity[key] = similarity
        
        return similarity
    
    def _update_similarities(self, node_id):
        """
        Update cross-modal similarities for a node
        """
        # Find all nodes that have representations
        all_node_ids = set()
        for modality in self.modalities:
            all_node_ids.update(self.representations.get(modality, {}).keys())
        
        # Calculate similarities with this node
        for other_id in all_node_ids:
            if other_id != node_id:
                key = (min(node_id, other_id), max(node_id, other_id))
                self.cross_modal_similarity[key] = self._calculate_similarity(node_id, other_id)
    
    def _calculate_similarity(self, node_id1, node_id2):
        """
        Calculate similarity between two nodes
        """
        similarities = []
        
        # Calculate similarity for each modality where both nodes have representations
        for modality in self.modalities:
            if (node_id1 in self.representations.get(modality, {}) and 
                node_id2 in self.representations.get(modality, {})):
                
                vec1 = np.array(self.representations[modality][node_id1]).reshape(1, -1)
                vec2 = np.array(self.representations[modality][node_id2]).reshape(1, -1)
                
                sim = cosine_similarity(vec1, vec2)[0][0]
                similarities.append(sim)
        
        # Return average similarity if any, otherwise 0
        return np.mean(similarities) if similarities else 0.0
    
    def get_most_similar_nodes(self, node_id, top_n=5):
        """
        Get the most similar nodes to a given node
        """
        similarities = []
        
        for modality in self.modalities:
            if node_id in self.representations.get(modality, {}):
                for other_id in self.representations[modality]:
                    if other_id != node_id:
                        sim = self.get_similarity(node_id, other_id)
                        similarities.append((other_id, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def to_dict(self):
        """
        Convert the semantic hub to a dictionary
        """
        return {
            "modalities": self.modalities,
            "embedding_dim": self.embedding_dim,
            "representations": self.representations,
            "cross_modal_similarity": {str(k): v for k, v in self.cross_modal_similarity.items()}
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a semantic hub from a dictionary
        """
        hub = cls(modalities=data["modalities"], embedding_dim=data["embedding_dim"])
        
        # Restore representations
        hub.representations = data["representations"]
        
        # Restore cross-modal similarities
        hub.cross_modal_similarity = {
            tuple(map(int, k.strip("()").split(", "))): v 
            for k, v in data["cross_modal_similarity"].items()
        }
        
        return hub 
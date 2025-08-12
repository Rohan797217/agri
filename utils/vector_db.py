import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pickle
from datetime import datetime
import hashlib

class VectorDB:
    """Simple vector database for document storage and retrieval"""
    
    def __init__(self, db_path: str = "vector_db"):
        self.db_path = db_path
        self.documents = {}
        self.vectors = {}
        self.metadata = {}
        self.create_db_structure()
        self.load_database()
    
    def create_db_structure(self):
        """Create database directory structure"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            os.makedirs(os.path.join(self.db_path, "documents"), exist_ok=True)
            os.makedirs(os.path.join(self.db_path, "vectors"), exist_ok=True)
            os.makedirs(os.path.join(self.db_path, "metadata"), exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create database structure: {e}")
    
    def load_database(self):
        """Load existing database files"""
        try:
            # Load documents
            docs_file = os.path.join(self.db_path, "documents.json")
            if os.path.exists(docs_file):
                with open(docs_file, 'r', encoding='utf-8') as f:
                    self.documents = json.load(f)
            
            # Load metadata
            meta_file = os.path.join(self.db_path, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            # Load vectors (pickle format for numpy arrays)
            vectors_file = os.path.join(self.db_path, "vectors.pkl")
            if os.path.exists(vectors_file):
                with open(vectors_file, 'rb') as f:
                    self.vectors = pickle.load(f)
                    
        except Exception as e:
            print(f"Warning: Could not load existing database: {e}")
    
    def save_database(self):
        """Save database to disk"""
        try:
            # Save documents
            docs_file = os.path.join(self.db_path, "documents.json")
            with open(docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            
            # Save metadata
            meta_file = os.path.join(self.db_path, "metadata.json")
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            # Save vectors
            vectors_file = os.path.join(self.db_path, "vectors.pkl")
            with open(vectors_file, 'wb') as f:
                pickle.dump(self.vectors, f)
                
        except Exception as e:
            print(f"Warning: Could not save database: {e}")
    
    def generate_document_id(self, content: str, filename: str = "") -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5((content + filename).encode()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"doc_{timestamp}_{content_hash[:8]}"
    
    def create_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding for text (placeholder for real embeddings)"""
        try:
            # Simple bag-of-words embedding (replace with real embeddings in production)
            words = text.lower().split()
            
            # Create vocabulary from agricultural terms
            vocab = self.get_agricultural_vocabulary()
            
            # Create vector
            vector = np.zeros(len(vocab))
            
            for word in words:
                if word in vocab:
                    idx = vocab.index(word)
                    vector[idx] += 1
            
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector
            
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return np.random.rand(100)  # Fallback random vector
    
    def get_agricultural_vocabulary(self) -> List[str]:
        """Get agricultural vocabulary for embedding"""
        return [
            # Crops
            "rice", "wheat", "cotton", "sugarcane", "maize", "soybean", "groundnut",
            "onion", "potato", "tomato", "chilli", "turmeric", "cardamom",
            # Farming terms
            "irrigation", "fertilizer", "pesticide", "harvest", "planting", "sowing",
            "cultivation", "yield", "production", "crop", "field", "farm",
            # Weather
            "rain", "temperature", "humidity", "weather", "climate", "season",
            "monsoon", "drought", "flood", "frost",
            # Diseases
            "disease", "pest", "fungus", "infection", "virus", "bacteria",
            "blight", "rust", "rot", "wilt",
            # Market
            "price", "market", "sell", "buy", "rate", "profit", "cost",
            "mandi", "trader", "export", "import",
            # Finance
            "loan", "credit", "bank", "subsidy", "scheme", "insurance",
            "kisan", "pmkisan", "kcc", "mudra",
            # Policy
            "government", "policy", "scheme", "yojana", "ministry", "department",
            # Soil
            "soil", "nitrogen", "phosphorus", "potassium", "ph", "organic",
            "fertility", "erosion", "drainage",
            # Equipment
            "tractor", "plough", "harvester", "sprayer", "pump", "machinery",
            # General
            "farmer", "agriculture", "farming", "rural", "village", "land",
            "water", "seed", "growth", "plant", "tree"
        ]
    
    def add_document(self, content: str, filename: str = "", 
                    doc_type: str = "general", metadata: Dict = None) -> str:
        """Add document to vector database"""
        try:
            # Generate document ID
            doc_id = self.generate_document_id(content, filename)
            
            # Create embedding
            vector = self.create_embedding(content)
            
            # Store document
            self.documents[doc_id] = {
                "content": content,
                "filename": filename,
                "type": doc_type,
                "created_at": datetime.now().isoformat()
            }
            
            # Store vector
            self.vectors[doc_id] = vector
            
            # Store metadata
            self.metadata[doc_id] = {
                "doc_type": doc_type,
                "filename": filename,
                "content_length": len(content),
                "word_count": len(content.split()),
                "created_at": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Save to disk
            self.save_database()
            
            return doc_id
            
        except Exception as e:
            raise Exception(f"Failed to add document: {str(e)}")
    
    def search_similar(self, query: str, top_k: int = 5, 
                      doc_type: str = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if not self.vectors:
                return []
            
            # Create query embedding
            query_vector = self.create_embedding(query)
            
            # Calculate similarities
            similarities = []
            
            for doc_id, doc_vector in self.vectors.items():
                # Filter by document type if specified
                if doc_type and self.metadata.get(doc_id, {}).get("doc_type") != doc_type:
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(query_vector, doc_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                )
                
                similarities.append({
                    "doc_id": doc_id,
                    "similarity": float(similarity),
                    "document": self.documents[doc_id],
                    "metadata": self.metadata.get(doc_id, {})
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from database"""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
            if doc_id in self.vectors:
                del self.vectors[doc_id]
            if doc_id in self.metadata:
                del self.metadata[doc_id]
            
            self.save_database()
            return True
            
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_all_documents(self, doc_type: str = None) -> List[Dict]:
        """Get all documents, optionally filtered by type"""
        try:
            results = []
            
            for doc_id, document in self.documents.items():
                if doc_type and self.metadata.get(doc_id, {}).get("doc_type") != doc_type:
                    continue
                
                results.append({
                    "doc_id": doc_id,
                    "document": document,
                    "metadata": self.metadata.get(doc_id, {})
                })
            
            return results
            
        except Exception as e:
            print(f"Error getting documents: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            doc_types = {}
            total_content_length = 0
            
            for doc_id, metadata in self.metadata.items():
                doc_type = metadata.get("doc_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                total_content_length += metadata.get("content_length", 0)
            
            return {
                "total_documents": len(self.documents),
                "document_types": doc_types,
                "total_content_length": total_content_length,
                "average_document_length": total_content_length / len(self.documents) if self.documents else 0,
                "database_size_mb": self.estimate_database_size()
            }
            
        except Exception as e:
            return {"error": f"Could not get database stats: {e}"}
    
    def estimate_database_size(self) -> float:
        """Estimate database size in MB"""
        try:
            total_size = 0
            
            # Size of documents
            for content in self.documents.values():
                total_size += len(str(content))
            
            # Size of vectors
            for vector in self.vectors.values():
                total_size += vector.nbytes
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0
    
    def clear_database(self) -> bool:
        """Clear all data from database"""
        try:
            self.documents = {}
            self.vectors = {}
            self.metadata = {}
            self.save_database()
            return True
            
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False

import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from backend.rag.embeddings import HFEmbeddings
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """
    Vector database for storing and retrieving FAQ embeddings.
    """
    
    def __init__(
        self,
        collection_name: str = "clinic_faq",
        persist_directory: str = "./vectordb"
    ):
        """
        Initialize vector store with ChromaDB.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        self.embeddings = HFEmbeddings()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name
            )
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
        """
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(documents)
        
        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of documents with metadata and scores
        """
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0,
                    'id': results['ids'][0][i]
                })
        
        return formatted_results
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def initialize_from_json(self, json_path: str):
        """
        Initialize vector store from a JSON file.
        
        Args:
            json_path: Path to the JSON file containing FAQ data
        """
        # Check if collection already has data
        if self.collection.count() > 0:
            print("Collection already initialized with data")
            return
        
        # Load JSON data
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Creating empty collection.")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract documents and metadata
        documents = []
        metadatas = []
        ids = []
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # If it's a dictionary with FAQ structure
            if 'faqs' in data:
                faqs = data['faqs']
            elif 'questions' in data:
                faqs = data['questions']
            else:
                # Treat each key-value as a FAQ
                faqs = [{'question': k, 'answer': v} for k, v in data.items()]
            
            for idx, faq in enumerate(faqs):
                if isinstance(faq, dict):
                    question = faq.get('question', '')
                    answer = faq.get('answer', '')
                    category = faq.get('category', 'general')
                    
                    # Combine question and answer for better retrieval
                    doc_text = f"Question: {question}\nAnswer: {answer}"
                    documents.append(doc_text)
                    metadatas.append({
                        'question': question,
                        'answer': answer,
                        'category': category
                    })
                    ids.append(f"faq_{idx}")
        
        elif isinstance(data, list):
            # If it's a list of FAQs
            for idx, faq in enumerate(data):
                if isinstance(faq, dict):
                    question = faq.get('question', '')
                    answer = faq.get('answer', '')
                    category = faq.get('category', 'general')
                    
                    doc_text = f"Question: {question}\nAnswer: {answer}"
                    documents.append(doc_text)
                    metadatas.append({
                        'question': question,
                        'answer': answer,
                        'category': category
                    })
                    ids.append(f"faq_{idx}")
        
        if documents:
            self.add_documents(documents, metadatas, ids)
            print(f"Initialized vector store with {len(documents)} FAQs")
        else:
            print("No valid FAQ data found in JSON file")


def get_vector_store() -> VectorStore:
    """
    Factory function to get vector store instance.
    
    Returns:
        Configured VectorStore instance
    """
    collection_name = os.getenv("COLLECTION_NAME", "clinic_faq")
    persist_directory = os.getenv("VECTOR_DB_PATH", "./vectordb")
    
    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
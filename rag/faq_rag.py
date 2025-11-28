import os
from typing import List, Dict, Any, Tuple
from groq import Groq
from backend.rag.vector_store import VectorStore
from dotenv import load_dotenv

load_dotenv()


class FAQRAG:
    """
    FAQ Retrieval-Augmented Generation system.
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize FAQ RAG system.
        
        Args:
            vector_store: Vector store instance for retrieval
        """
        self.vector_store = vector_store
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=self.groq_api_key)
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
    
    def retrieve_context(
        self,
        query: str,
        k: int = 3,
        min_similarity: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve relevant context from vector store.
        
        Args:
            query: User's question
            k: Number of documents to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            Tuple of (retrieved documents, formatted context string)
        """
        # Search for similar documents
        results = self.vector_store.similarity_search(query, k=k)
        
        # Filter by similarity threshold
        # Note: ChromaDB returns distances, lower is better (cosine distance)
        # We convert to similarity: similarity = 1 - distance
        filtered_results = []
        for result in results:
            similarity = 1 - result['distance']
            if similarity >= min_similarity:
                result['similarity'] = similarity
                filtered_results.append(result)
        
        # Format context
        if not filtered_results:
            return [], "No relevant information found in the knowledge base."
        
        context_parts = []
        for idx, result in enumerate(filtered_results, 1):
            metadata = result['metadata']
            context_parts.append(
                f"[Source {idx}]\n"
                f"Q: {metadata.get('question', 'N/A')}\n"
                f"A: {metadata.get('answer', 'N/A')}\n"
            )
        
        context = "\n".join(context_parts)
        return filtered_results, context
    
    def generate_answer(
        self,
        question: str,
        context: str,
        model: str = "openai/gpt-oss-120b",
        session_id: str = None
    ) -> str:
        """
        Generate answer using retrieved context.
        
        Args:
            question: User's question
            context: Retrieved context
            model: LLM model to use
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Generated answer
        """
        # Build conversation history
        if session_id and session_id in self.conversation_history:
            messages = self.conversation_history[session_id].copy()
        else:
            messages = []
        
        # System prompt
        system_prompt = """You are a helpful medical clinic assistant. Your role is to answer questions about the clinic using ONLY the provided context.

Guidelines:
- Answer questions accurately based on the context provided
- If the context doesn't contain relevant information, say so clearly
- Be friendly and professional
- Keep answers concise but complete
- Do not make up or infer information not in the context
- If asked about previous topics in the conversation, refer to the conversation history"""

        if not messages:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add context and question
        user_message = f"""Context from clinic knowledge base:
{context}

Question: {question}

Please provide a helpful answer based on the context above."""
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # Update conversation history
            if session_id:
                if session_id not in self.conversation_history:
                    self.conversation_history[session_id] = [
                        {"role": "system", "content": system_prompt}
                    ]
                
                self.conversation_history[session_id].append({
                    "role": "user",
                    "content": question
                })
                self.conversation_history[session_id].append({
                    "role": "assistant",
                    "content": answer
                })
                
                # Keep only last 10 messages (5 turns) plus system message
                if len(self.conversation_history[session_id]) > 11:
                    self.conversation_history[session_id] = (
                        [self.conversation_history[session_id][0]] +
                        self.conversation_history[session_id][-10:]
                    )
            
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def ask(
        self,
        question: str,
        model: str = "openai/gpt-oss-120b",
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.
        
        Args:
            question: User's question
            model: LLM model to use
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Dictionary with answer, sources, and confidence
        """
        # Retrieve relevant context
        results, context = self.retrieve_context(question)
        
        # Generate answer
        answer = self.generate_answer(question, context, model, session_id)
        
        # Calculate confidence
        if results:
            avg_similarity = sum(r['similarity'] for r in results) / len(results)
            confidence = min(avg_similarity, 1.0)
        else:
            confidence = 0.0
        
        # Extract source questions
        sources = [r['metadata'].get('question', 'Unknown') for r in results]
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'context_used': bool(results)
        }
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]


def get_faq_rag(vector_store: VectorStore) -> FAQRAG:
    """
    Factory function to get FAQ RAG instance.
    
    Args:
        vector_store: Vector store instance
        
    Returns:
        Configured FAQRAG instance
    """
    return FAQRAG(vector_store)
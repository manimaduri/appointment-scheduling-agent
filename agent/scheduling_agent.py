import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from groq import Groq
from backend.agent.prompts import SYSTEM_PROMPT
from backend.tools.availability_tool import AvailabilityTool
from backend.tools.booking_tool import BookingTool
from backend.rag.faq_rag import FAQRAG
from dotenv import load_dotenv

load_dotenv()


class SchedulingAgent:
    """
    Conversational agent for appointment scheduling with tool use.
    """
    
    def __init__(
        self,
        faq_rag: FAQRAG,
        availability_tool: AvailabilityTool,
        booking_tool: BookingTool
    ):
        """
        Initialize scheduling agent.
        
        Args:
            faq_rag: FAQ RAG system
            availability_tool: Tool for checking availability
            booking_tool: Tool for booking appointments
        """
        self.faq_rag = faq_rag
        self.availability_tool = availability_tool
        self.booking_tool = booking_tool
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        self.client = Groq(api_key=self.groq_api_key)
        
        # Session management
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}
    
    def _get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Get or create session message history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
        return self.sessions[session_id]
    
    def _add_message(self, session_id: str, role: str, content: str):
        """Add message to session history."""
        messages = self._get_session_messages(session_id)
        messages.append({"role": role, "content": content})
        
        # Keep only last 20 messages plus system message
        if len(messages) > 21:
            self.sessions[session_id] = [messages[0]] + messages[-20:]
    
    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get tools schema for function calling."""
        return [
            self.availability_tool.get_tool_description(),
            self.booking_tool.get_tool_description()
        ]
    
    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool and return results."""
        if tool_name == "check_availability":
            result = await self.availability_tool.check_availability(**tool_args)
            return result
        
        elif tool_name == "book_appointment":
            result = await self.booking_tool.book_appointment(**tool_args)
            return result
        
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    async def _handle_tool_calls(
        self,
        response: Any,
        session_id: str,
        model: str
    ) -> str:
        """Handle tool calls from LLM response."""
        tool_calls = response.choices[0].message.tool_calls
        
        if not tool_calls:
            return response.choices[0].message.content
        
        # Execute each tool call
        messages = self._get_session_messages(session_id)
        
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in tool_calls
            ]
        })
        
        # Execute tools and add results
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            result = await self._execute_tool(function_name, function_args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(result)
            })
        
        # Get final response with tool results
        final_response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return final_response.choices[0].message.content
    
    async def _check_faq_intent(self, message: str, model: str) -> Optional[str]:
        """Check if message is a FAQ question and answer it."""
        # Simple heuristic: if message is a question and doesn't contain booking keywords
        booking_keywords = [
            'book', 'schedule', 'appointment', 'reserve', 
            'available', 'availability', 'slot'
        ]
        
        is_question = any(q in message.lower() for q in ['what', 'where', 'when', 'how', 'who', 'why', '?'])
        has_booking_intent = any(kw in message.lower() for kw in booking_keywords)
        
        if is_question and not has_booking_intent:
            # Try to answer with FAQ
            faq_result = self.faq_rag.ask(message, model=model)
            
            if faq_result['confidence'] > 0.5:
                return faq_result['answer']
        
        return None
    
    async def chat(
        self,
        message: str,
        session_id: str,
        model: str = "openai/gpt-oss-120b"
    ) -> str:
        """
        Process a chat message and return response.
        
        Args:
            message: User's message
            session_id: Session identifier
            model: LLM model to use
            
        Returns:
            Agent's response
        """
        try:
            # Add current date context
            current_date = datetime.now().strftime("%Y-%m-%d")
            message_with_context = f"[Today's date: {current_date}]\n\nUser message: {message}"
            
            # Check if it's a FAQ question first
            faq_answer = await self._check_faq_intent(message, model)
            if faq_answer:
                self._add_message(session_id, "user", message)
                self._add_message(session_id, "assistant", faq_answer)
                return faq_answer
            
            # Add user message to history
            self._add_message(session_id, "user", message_with_context)
            
            # Get messages for this session
            messages = self._get_session_messages(session_id)
            
            # Get response from LLM with tools
            # response = self.client.chat.completions.create(
            #     model=model,
            #     messages=messages,
            #     tools=self._get_tools_schema(),
            #     tool_choice="auto",
            #     temperature=0.7,
            #     max_tokens=1000
            # )
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self._get_tools_schema(),
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )

            # Handle tool calls if any
            if response.choices[0].message.tool_calls:
                final_response = await self._handle_tool_calls(
                    response, session_id, model
                )
            else:
                final_response = response.choices[0].message.content
            
            # Add assistant response to history
            self._add_message(session_id, "assistant", final_response)
            
            return final_response
            
        except Exception as e:
            error_message = f"I apologize, but I encountered an error: {str(e)}"
            print(f"Error in chat: {e}")
            return error_message
    
    def clear_session(self, session_id: str):
        """Clear session history."""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        return {
            "session_id": session_id,
            "message_count": len(self.sessions.get(session_id, [])) - 1,  # Exclude system message
            "session_data": self.session_data.get(session_id, {})
        }


def get_scheduling_agent(
    faq_rag: FAQRAG,
    availability_tool: AvailabilityTool,
    booking_tool: BookingTool
) -> SchedulingAgent:
    """
    Factory function to create scheduling agent.
    
    Args:
        faq_rag: FAQ RAG system
        availability_tool: Availability checking tool
        booking_tool: Booking tool
        
    Returns:
        Configured SchedulingAgent instance
    """
    return SchedulingAgent(faq_rag, availability_tool, booking_tool)
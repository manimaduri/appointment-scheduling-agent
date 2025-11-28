import httpx
from typing import Dict, Any, Optional
from datetime import datetime


class AvailabilityTool:
    """
    Tool for checking appointment availability via Calendly API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize availability tool.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/calendly/availability"
    
    async def check_availability(
        self,
        date: str,
        appointment_type: str,
        doctor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check availability for a specific date and appointment type.
        
        Args:
            date: Date in YYYY-MM-DD format
            appointment_type: Type of appointment (Consultation, Follow-up, etc.)
            doctor: Optional specific doctor name
            
        Returns:
            Dictionary with availability information
        """
        try:
            params = {
                "date": date,
                "appointment_type": appointment_type
            }
            
            if doctor:
                params["doctor"] = doctor
            
            async with httpx.AsyncClient() as client:
                response = await client.get(self.endpoint, params=params)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "error": f"Failed to check availability: {response.status_code}",
                        "details": response.text
                    }
                    
        except Exception as e:
            return {
                "error": f"Error checking availability: {str(e)}"
            }
    
    def format_availability_response(self, data: Dict[str, Any]) -> str:
        """
        Format availability response for display to user.
        
        Args:
            data: Raw availability data
            
        Returns:
            Formatted string for user
        """
        if "error" in data:
            return f"Error: {data['error']}"
        
        date = data.get('date', 'Unknown')
        appointment_type = data.get('appointment_type', 'Unknown')
        slots = data.get('slots', [])
        message = data.get('message', '')
        
        if message:
            return message
        
        if not slots:
            return f"No availability found for {appointment_type} on {date}."
        
        # Group slots by doctor
        doctor_slots = {}
        for slot in slots:
            if slot['available']:
                doctor = slot['doctor']
                if doctor not in doctor_slots:
                    doctor_slots[doctor] = []
                doctor_slots[doctor].append(slot['time'])
        
        if not doctor_slots:
            return f"All slots are booked for {appointment_type} on {date}. Please try another date."
        
        # Format response
        response_parts = [f"Available slots for {appointment_type} on {date}:\n"]
        
        for doctor, times in doctor_slots.items():
            response_parts.append(f"\n{doctor}:")
            response_parts.append(", ".join(times))
        
        return "\n".join(response_parts)
    
    def get_tool_description(self) -> Dict[str, Any]:
        """
        Get tool description for LLM function calling.
        
        Returns:
            Tool description in JSON schema format
        """
        return {
            "type": "function",
            "function": {
                "name": "check_availability",
                "description": "Check available appointment slots for a specific date and appointment type. Use this when the user wants to know available times for booking.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format (e.g., 2024-12-15)"
                        },
                        "appointment_type": {
                            "type": "string",
                            "enum": ["Consultation", "Follow-up", "Check-up", "Vaccination"],
                            "description": "Type of appointment"
                        },
                        "doctor": {
                            "type": "string",
                            "description": "Specific doctor name (optional)"
                        }
                    },
                    "required": ["date", "appointment_type"]
                }
            }
        }


def get_availability_tool(base_url: str = "http://localhost:8000") -> AvailabilityTool:
    """
    Factory function to create availability tool.
    
    Args:
        base_url: Base URL of the API server
        
    Returns:
        Configured AvailabilityTool instance
    """
    return AvailabilityTool(base_url)
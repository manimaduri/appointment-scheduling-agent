import httpx
from typing import Dict, Any, Optional


class BookingTool:
    """
    Tool for booking appointments via Calendly API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize booking tool.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/calendly/book"
    
    async def book_appointment(
        self,
        patient_name: str,
        email: str,
        phone: str,
        date: str,
        time: str,
        appointment_type: str,
        doctor: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Book an appointment.
        
        Args:
            patient_name: Patient's full name
            email: Patient's email address
            phone: Patient's phone number
            date: Appointment date in YYYY-MM-DD format
            time: Appointment time in HH:MM format
            appointment_type: Type of appointment
            doctor: Doctor name
            notes: Optional notes
            
        Returns:
            Dictionary with booking confirmation
        """
        try:
            booking_data = {
                "patient_name": patient_name,
                "email": email,
                "phone": phone,
                "date": date,
                "time": time,
                "appointment_type": appointment_type,
                "doctor": doctor
            }
            
            if notes:
                booking_data["notes"] = notes
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.endpoint, json=booking_data)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "success": False,
                        "error": f"Failed to book appointment: {response.status_code}",
                        "details": response.text
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Error booking appointment: {str(e)}"
            }
    
    def format_booking_response(self, data: Dict[str, Any]) -> str:
        """
        Format booking response for display to user.
        
        Args:
            data: Raw booking response data
            
        Returns:
            Formatted string for user
        """
        if not data.get('success', False):
            error_msg = data.get('message') or data.get('error', 'Unknown error')
            return f"Booking failed: {error_msg}"
        
        booking_id = data.get('booking_id', 'N/A')
        message = data.get('message', 'Appointment booked successfully')
        details = data.get('appointment_details', {})
        
        response_parts = [
            f"✅ {message}",
            f"\nBooking ID: {booking_id}"
        ]
        
        if details:
            response_parts.extend([
                f"Patient: {details.get('patient_name', 'N/A')}",
                f"Date: {details.get('date', 'N/A')}",
                f"Time: {details.get('time', 'N/A')}",
                f"Type: {details.get('appointment_type', 'N/A')}",
                f"Doctor: {details.get('doctor', 'N/A')}",
                f"Duration: {details.get('duration_minutes', 'N/A')} minutes"
            ])
            
            if details.get('notes'):
                response_parts.append(f"Notes: {details['notes']}")
        
        response_parts.append("\n📧 A confirmation email will be sent to your email address.")
        
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
                "name": "book_appointment",
                "description": "Book an appointment with the clinic. Use this when the user wants to confirm a booking with all required information provided.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_name": {
                            "type": "string",
                            "description": "Patient's full name"
                        },
                        "email": {
                            "type": "string",
                            "description": "Patient's email address"
                        },
                        "phone": {
                            "type": "string",
                            "description": "Patient's phone number"
                        },
                        "date": {
                            "type": "string",
                            "description": "Appointment date in YYYY-MM-DD format"
                        },
                        "time": {
                            "type": "string",
                            "description": "Appointment time in HH:MM format (24-hour)"
                        },
                        "appointment_type": {
                            "type": "string",
                            "enum": ["Consultation", "Follow-up", "Check-up", "Vaccination"],
                            "description": "Type of appointment"
                        },
                        "doctor": {
                            "type": "string",
                            "description": "Doctor name"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Optional notes or special requirements"
                        }
                    },
                    "required": [
                        "patient_name",
                        "email",
                        "phone",
                        "date",
                        "time",
                        "appointment_type",
                        "doctor"
                    ]
                }
            }
        }


def get_booking_tool(base_url: str = "http://localhost:8000") -> BookingTool:
    """
    Factory function to create booking tool.
    
    Args:
        base_url: Base URL of the API server
        
    Returns:
        Configured BookingTool instance
    """
    return BookingTool(base_url)
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date, time
from enum import Enum


class AppointmentType(str, Enum):
    CONSULTATION = "Consultation"
    FOLLOWUP = "Follow-up"
    CHECKUP = "Check-up"
    VACCINATION = "Vaccination"


class ModelChoice(str, Enum):
    GPT_OSS_120B = "openai/gpt-oss-120b"
    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick-17b-128e-instruct"
    LLAMA_4_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"
    GPT_OSS_20B = "openai/gpt-oss-20b"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's message")
    session_id: str = Field(..., description="Unique session identifier")
    model: Optional[ModelChoice] = Field(
        default=ModelChoice.GPT_OSS_120B,
        description="LLM model to use"
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    model_used: str = Field(..., description="Model that generated the response")


class FAQRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    model: Optional[ModelChoice] = Field(
        default=ModelChoice.GPT_OSS_120B,
        description="LLM model to use"
    )


class FAQResponse(BaseModel):
    answer: str = Field(..., description="Answer to the question")
    sources: List[str] = Field(default=[], description="Source documents used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model_used: str = Field(..., description="Model that generated the answer")


class AvailabilityRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    appointment_type: AppointmentType = Field(..., description="Type of appointment")
    doctor: Optional[str] = Field(None, description="Specific doctor name")

    @validator('date')
    def validate_date(cls, v):
        try:
            date_obj = datetime.strptime(v, '%Y-%m-%d').date()
            if date_obj < datetime.now().date():
                raise ValueError('Date cannot be in the past')
            return v
        except ValueError as e:
            if 'Date cannot be in the past' in str(e):
                raise e
            raise ValueError('Date must be in YYYY-MM-DD format')


class TimeSlot(BaseModel):
    time: str = Field(..., description="Time in HH:MM format")
    available: bool = Field(..., description="Whether slot is available")
    doctor: str = Field(..., description="Doctor name")
    duration_minutes: int = Field(..., description="Appointment duration")


class AvailabilityResponse(BaseModel):
    date: str = Field(..., description="Requested date")
    appointment_type: str = Field(..., description="Appointment type")
    slots: List[TimeSlot] = Field(..., description="Available time slots")
    message: Optional[str] = Field(None, description="Additional information")


class BookingRequest(BaseModel):
    patient_name: str = Field(..., min_length=1, description="Patient's full name")
    email: EmailStr = Field(..., description="Patient's email")
    phone: str = Field(..., description="Patient's phone number")
    date: str = Field(..., description="Appointment date in YYYY-MM-DD format")
    time: str = Field(..., description="Appointment time in HH:MM format")
    appointment_type: AppointmentType = Field(..., description="Type of appointment")
    doctor: str = Field(..., description="Doctor name")
    notes: Optional[str] = Field(None, description="Additional notes")

    @validator('date')
    def validate_date(cls, v):
        try:
            date_obj = datetime.strptime(v, '%Y-%m-%d').date()
            if date_obj < datetime.now().date():
                raise ValueError('Date cannot be in the past')
            return v
        except ValueError as e:
            if 'Date cannot be in the past' in str(e):
                raise e
            raise ValueError('Date must be in YYYY-MM-DD format')

    @validator('time')
    def validate_time(cls, v):
        try:
            datetime.strptime(v, '%H:%M')
            return v
        except ValueError:
            raise ValueError('Time must be in HH:MM format')

    @validator('phone')
    def validate_phone(cls, v):
        # Remove common formatting characters
        cleaned = ''.join(c for c in v if c.isdigit() or c == '+')
        if len(cleaned) < 10:
            raise ValueError('Phone number must have at least 10 digits')
        return v


class BookingResponse(BaseModel):
    success: bool = Field(..., description="Whether booking was successful")
    booking_id: Optional[str] = Field(None, description="Unique booking identifier")
    message: str = Field(..., description="Confirmation or error message")
    appointment_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Details of the booked appointment"
    )


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now)
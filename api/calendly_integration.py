from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
import json
import os
from backend.models.schemas import (
    AvailabilityRequest,
    AvailabilityResponse,
    BookingRequest,
    BookingResponse,
    TimeSlot,
    AppointmentType,
    ErrorResponse
)

router = APIRouter(prefix="/api/calendly", tags=["Calendly Integration"])


# Mock database for bookings
bookings_db: Dict[str, Dict] = {}

# Mock doctor schedules
DOCTOR_SCHEDULES = {
    "Dr. Smith": {
        "available_days": [0, 1, 2, 3, 4],  # Monday to Friday
        "hours": {
            "start": "09:00",
            "end": "17:00"
        },
        "lunch_break": {
            "start": "12:00",
            "end": "13:00"
        }
    },
    "Dr. Johnson": {
        "available_days": [0, 1, 2, 3, 4],
        "hours": {
            "start": "10:00",
            "end": "18:00"
        },
        "lunch_break": {
            "start": "13:00",
            "end": "14:00"
        }
    },
    "Dr. Williams": {
        "available_days": [1, 2, 3, 4],  # Tuesday to Friday
        "hours": {
            "start": "09:00",
            "end": "16:00"
        },
        "lunch_break": {
            "start": "12:30",
            "end": "13:30"
        }
    }
}

# Appointment type durations (in minutes)
APPOINTMENT_DURATIONS = {
    AppointmentType.CONSULTATION: 30,
    AppointmentType.FOLLOWUP: 15,
    AppointmentType.CHECKUP: 20,
    AppointmentType.VACCINATION: 10
}


def load_doctor_schedule_from_file():
    """Load doctor schedule from JSON file if available."""
    schedule_file = os.path.join("data", "doctor_schedule.json")
    if os.path.exists(schedule_file):
        try:
            with open(schedule_file, 'r') as f:
                data = json.load(f)
                # Merge with default schedules
                if 'doctors' in data:
                    DOCTOR_SCHEDULES.update(data['doctors'])
                if 'durations' in data:
                    for apt_type, duration in data['durations'].items():
                        if hasattr(AppointmentType, apt_type.upper().replace('-', '')):
                            APPOINTMENT_DURATIONS[
                                AppointmentType[apt_type.upper().replace('-', '')]
                            ] = duration
        except Exception as e:
            print(f"Error loading doctor schedule: {e}")


# Load schedule on module import
load_doctor_schedule_from_file()


def generate_time_slots(
    start_time: str,
    end_time: str,
    duration: int,
    lunch_start: str = None,
    lunch_end: str = None
) -> List[str]:
    """
    Generate time slots for a given time range.
    
    Args:
        start_time: Start time in HH:MM format
        end_time: End time in HH:MM format
        duration: Duration of each slot in minutes
        lunch_start: Lunch break start time
        lunch_end: Lunch break end time
        
    Returns:
        List of time strings in HH:MM format
    """
    slots = []
    current = datetime.strptime(start_time, "%H:%M")
    end = datetime.strptime(end_time, "%H:%M")
    
    lunch_start_dt = datetime.strptime(lunch_start, "%H:%M") if lunch_start else None
    lunch_end_dt = datetime.strptime(lunch_end, "%H:%M") if lunch_end else None
    
    while current < end:
        # Check if in lunch break
        if lunch_start_dt and lunch_end_dt:
            if lunch_start_dt <= current < lunch_end_dt:
                current += timedelta(minutes=duration)
                continue
        
        # Check if slot would extend past end time
        slot_end = current + timedelta(minutes=duration)
        if slot_end <= end:
            slots.append(current.strftime("%H:%M"))
        
        current += timedelta(minutes=duration)
    
    return slots


def is_slot_booked(date: str, time: str, doctor: str) -> bool:
    """Check if a time slot is already booked."""
    for booking in bookings_db.values():
        if (booking['date'] == date and 
            booking['time'] == time and 
            booking['doctor'] == doctor):
            return True
    return False


@router.get("/availability", response_model=AvailabilityResponse)
async def get_availability(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    appointment_type: AppointmentType = Query(..., description="Type of appointment"),
    doctor: Optional[str] = Query(None, description="Specific doctor (optional)")
):
    """
    Get available appointment slots for a given date and appointment type.
    
    Returns available time slots with doctor information.
    """
    try:
        # Validate date format
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        
        # Check if date is in the past
        if date_obj < datetime.now().date():
            raise HTTPException(
                status_code=400,
                detail="Cannot book appointments in the past"
            )
        
        # Get day of week (0 = Monday)
        day_of_week = date_obj.weekday()
        
        # Get appointment duration
        duration = APPOINTMENT_DURATIONS.get(appointment_type, 30)
        
        # Generate slots for each doctor (or specific doctor)
        all_slots = []
        doctors_to_check = [doctor] if doctor else list(DOCTOR_SCHEDULES.keys())
        
        for doc_name in doctors_to_check:
            if doc_name not in DOCTOR_SCHEDULES:
                continue
            
            schedule = DOCTOR_SCHEDULES[doc_name]
            
            # Check if doctor works on this day
            if day_of_week not in schedule['available_days']:
                continue
            
            # Generate time slots
            time_slots = generate_time_slots(
                schedule['hours']['start'],
                schedule['hours']['end'],
                duration,
                schedule['lunch_break']['start'],
                schedule['lunch_break']['end']
            )
            
            # Check availability for each slot
            for time_slot in time_slots:
                available = not is_slot_booked(date, time_slot, doc_name)
                all_slots.append(
                    TimeSlot(
                        time=time_slot,
                        available=available,
                        doctor=doc_name,
                        duration_minutes=duration
                    )
                )
        
        # Check if any slots available
        message = None
        if not all_slots:
            message = f"No availability found for {date}. Please try another date."
        elif not any(slot.available for slot in all_slots):
            message = "All slots are currently booked. Please try another date."
        
        return AvailabilityResponse(
            date=date,
            appointment_type=appointment_type.value,
            slots=all_slots,
            message=message
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving availability: {str(e)}"
        )


@router.post("/book", response_model=BookingResponse)
async def book_appointment(booking: BookingRequest):
    """
    Book an appointment with the clinic.
    
    Creates a booking and returns confirmation details.
    """
    try:
        # Validate date
        date_obj = datetime.strptime(booking.date, "%Y-%m-%d").date()
        if date_obj < datetime.now().date():
            return BookingResponse(
                success=False,
                message="Cannot book appointments in the past"
            )
        
        # Check if doctor exists
        if booking.doctor not in DOCTOR_SCHEDULES:
            return BookingResponse(
                success=False,
                message=f"Doctor '{booking.doctor}' not found"
            )
        
        # Check if doctor works on this day
        day_of_week = date_obj.weekday()
        schedule = DOCTOR_SCHEDULES[booking.doctor]
        
        if day_of_week not in schedule['available_days']:
            return BookingResponse(
                success=False,
                message=f"{booking.doctor} does not work on this day"
            )
        
        # Check if time is within working hours
        time_obj = datetime.strptime(booking.time, "%H:%M").time()
        start_time = datetime.strptime(schedule['hours']['start'], "%H:%M").time()
        end_time = datetime.strptime(schedule['hours']['end'], "%H:%M").time()
        
        if not (start_time <= time_obj < end_time):
            return BookingResponse(
                success=False,
                message=f"Selected time is outside {booking.doctor}'s working hours ({schedule['hours']['start']} - {schedule['hours']['end']})"
            )
        
        # Check if slot is already booked
        if is_slot_booked(booking.date, booking.time, booking.doctor):
            return BookingResponse(
                success=False,
                message="This time slot is already booked. Please choose another time."
            )
        
        # Create booking
        booking_id = str(uuid.uuid4())
        booking_data = {
            "booking_id": booking_id,
            "patient_name": booking.patient_name,
            "email": booking.email,
            "phone": booking.phone,
            "date": booking.date,
            "time": booking.time,
            "appointment_type": booking.appointment_type.value,
            "doctor": booking.doctor,
            "notes": booking.notes,
            "duration_minutes": APPOINTMENT_DURATIONS.get(
                booking.appointment_type, 30
            ),
            "status": "confirmed",
            "created_at": datetime.now().isoformat()
        }
        
        bookings_db[booking_id] = booking_data
        
        return BookingResponse(
            success=True,
            booking_id=booking_id,
            message=f"Appointment successfully booked with {booking.doctor} on {booking.date} at {booking.time}",
            appointment_details=booking_data
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error booking appointment: {str(e)}"
        )


@router.get("/bookings/{booking_id}")
async def get_booking(booking_id: str):
    """Get details of a specific booking."""
    if booking_id not in bookings_db:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    return bookings_db[booking_id]


@router.delete("/bookings/{booking_id}")
async def cancel_booking(booking_id: str):
    """Cancel a booking."""
    if booking_id not in bookings_db:
        raise HTTPException(status_code=404, detail="Booking not found")
    
    booking = bookings_db[booking_id]
    booking['status'] = 'cancelled'
    
    return {
        "success": True,
        "message": f"Booking {booking_id} has been cancelled",
        "booking": booking
    }
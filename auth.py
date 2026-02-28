import re
import secrets
import string
import bcrypt

def validate_password(password: str) -> bool:
    """
    Validates that a password is at least 8 characters long and contains
    at least one lowercase letter, one uppercase letter, one digit, and one special character.
    """
    if len(password) < 8:
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[0-9]', password):
        return False
    if not re.search(r'[@$!%*?&#]', password):
        return False
    return True

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """Verifies a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def generate_otp() -> str:
    """Generates a 6-digit OTP."""
    return ''.join(secrets.choice(string.digits) for _ in range(6))

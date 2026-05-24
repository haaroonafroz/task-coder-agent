"""Email validation module providing utilities for validating email addresses."""

import re

_EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')


def validate_email(address: str) -> bool:
    """Validate an email address.

    Args:
        address: The email address string to validate.

    Returns:
        True if the address is valid, False otherwise.
    """
    if not address or not isinstance(address, str):
        return False
    return bool(_EMAIL_PATTERN.match(address))

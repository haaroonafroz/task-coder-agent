import re

def validate_email(address: str) -> bool:
    if not isinstance(address, str) or not address:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, address):
        if '..' in address or address.startswith('.') or address.endswith('.'):
            return False
        return True
    return False
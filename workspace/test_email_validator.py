import pytest
from email_validator import validate_email

def test_valid_email():
    assert validate_email('test@example.com') is True

def test_invalid_email():
    assert validate_email('invalid-email') is False
    assert validate_email('') is False
    assert validate_email(None) is False

def test_edge_cases():
    assert validate_email('test..test@example.com') is False
    assert validate_email('.test@example.com') is False
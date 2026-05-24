import pytest
from validator.email import validate_email


class TestValidateEmail:
    """Unit tests for the email validator module."""

    # --- Valid Email Cases ---
    def test_valid_standard_email(self):
        assert validate_email("user@example.com") is True

    def test_valid_email_with_dots_and_plus(self):
        assert validate_email("user.name+tag@example.co.uk") is True

    def test_valid_minimal_email(self):
        assert validate_email("a@b.cd") is True

    def test_valid_email_with_subdomain(self):
        assert validate_email("test@mail.server.com") is True

    # --- Invalid Email Cases ---
    def test_invalid_no_at_symbol(self):
        assert validate_email("plaintext") is False

    def test_invalid_missing_username(self):
        assert validate_email("@example.com") is False

    def test_invalid_missing_domain(self):
        assert validate_email("username@") is False

    def test_invalid_short_tld(self):
        assert validate_email("user@domain.c") is False

    def test_invalid_multiple_at(self):
        assert validate_email("user@@domain.com") is False

    def test_invalid_spaces_in_local_part(self):
        assert validate_email("user name@domain.com") is False

    def test_invalid_trailing_dot_in_domain(self):
        assert validate_email("user@domain.com.") is False

    def test_invalid_leading_dot_in_domain(self):
        assert validate_email("user@.domain.com") is False

    # --- Type & Edge Cases ---
    def test_empty_string(self):
        assert validate_email("") is False

    def test_non_string_input_int(self):
        assert validate_email(123) is False

    def test_non_string_input_none(self):
        assert validate_email(None) is False

    def test_non_string_input_list(self):
        assert validate_email([]) is False

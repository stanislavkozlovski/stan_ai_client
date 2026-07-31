from __future__ import annotations

import pytest

JULY_31_DNS_DIAGNOSTIC = "failed to lookup address information: Try again"


@pytest.fixture
def july_31_dns_diagnostic() -> str:
    return JULY_31_DNS_DIAGNOSTIC

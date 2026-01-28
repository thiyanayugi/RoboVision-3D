"""Shared test fixtures and configuration."""

import pytest
import os


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary for testing."""
    return {
        "model": "yolov8n.pt",
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45,
        "device": "cpu",
    }

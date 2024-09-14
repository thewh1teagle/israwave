"""
Provide a way to enable logging by setting LOG_LEVEL environment variable
"""
import logging
import os

def setup_logging():
    # Set default logging level to WARNING if LOG_LEVEL is not set
    log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
    
    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING))
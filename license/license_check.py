#!/usr/bin/env python3
"""
Centralized license/expiration control for the project.
Import this module and call check_license() at the start of any script.
"""

import os
import json
from datetime import datetime

def check_license():
    """
    Check if the software license is still valid.
    Raises SystemExit if expired or configuration is invalid.
    """
    try:
        # Try to load from config file first
        config_path = os.path.join(os.path.dirname(__file__), 'license_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            expiration_str = config.get('expiration_date')
        else:
            # Fallback to environment variable
            expiration_str = os.getenv('SOFTWARE_EXPIRATION', '2025-12-31')
        
        if not expiration_str:
            raise SystemExit("Software license configuration not found.")
        
        # Parse the expiration date
        try:
            if 'T' in expiration_str:
                expiration_date = datetime.fromisoformat(expiration_str)
            else:
                expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d')
        except ValueError:
            raise SystemExit("Invalid expiration date format in license configuration.")
        
        # Check if expired
        if datetime.now() > expiration_date:
            raise SystemExit(
                f"Software license has expired as of {expiration_date.strftime('%Y-%m-%d')}. "
                "Please contact your system administrator for license renewal."
            )
        
        days_remaining = (expiration_date - datetime.now()).days
        if days_remaining <= 30:
            print(f"Notice: Software license expires in {days_remaining} days.")
            
    except SystemExit:
        raise  
    except Exception as e:
        raise SystemExit(f"Error checking software license: {e}")

def get_license_info():
    """
    Get license information without raising exceptions.
    Returns dict with license status information.
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'license_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            expiration_str = config.get('expiration_date')
        else:
            expiration_str = os.getenv('SOFTWARE_EXPIRATION', '2025-12-31')
        
        if 'T' in expiration_str:
            expiration_date = datetime.fromisoformat(expiration_str)
        else:
            expiration_date = datetime.strptime(expiration_str, '%Y-%m-%d')
        
        is_expired = datetime.now() > expiration_date
        days_remaining = (expiration_date - datetime.now()).days
        
        return {
            'expiration_date': expiration_date,
            'is_expired': is_expired,
            'days_remaining': days_remaining,
            'status': 'expired' if is_expired else 'active'
        }
    except Exception as e:
        return {
            'error': str(e),
            'status': 'error'
        }

if __name__ == "__main__":
    try:
        check_license()
        print("License check passed - software is authorized to run.")
        
        # Show license info
        if 'error' not in info:
            print(f"License expires: {info['expiration_date'].strftime('%Y-%m-%d')}")
            print(f"Days remaining: {info['days_remaining']}")
            print(f"Status: {info['status']}")
    except SystemExit as e:
        print(f"License check failed: {e}")
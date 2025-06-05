import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from src.config_manager import ConfigManager


class TestConfigManager:
    
    def test_init_with_valid_config(self):
        config_data = {"key1": "value1", "key2": "value2"}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            assert config_manager.settings == config_data
        finally:
            Path(config_file).unlink()
    
    def test_init_with_nonexistent_config(self):
        with pytest.raises(SystemExit):
            ConfigManager("nonexistent_config.json")
    
    def test_get_existing_key(self):
        config_data = {"existing_key": "existing_value"}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            result = config_manager.get("existing_key")
            assert result == "existing_value"
        finally:
            Path(config_file).unlink()
    
    def test_get_nonexistent_key_with_default(self):
        config_data = {"key1": "value1"}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            result = config_manager.get("nonexistent_key", "default_value")
            assert result == "default_value"
        finally:
            Path(config_file).unlink()
    
    def test_get_nonexistent_key_without_default(self):
        config_data = {"key1": "value1"}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            result = config_manager.get("nonexistent_key")
            assert result is None
        finally:
            Path(config_file).unlink()
    
    def test_get_directory_names(self):
        config_data = {
            "data_directory": "/path/to/data",
            "output_directory": "/path/to/output",
            "regular_key": "regular_value",
            "another_directory": "/path/to/another"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            directories = config_manager.get_directory_names()
            expected = ["/path/to/data", "/path/to/output", "/path/to/another"]
            assert set(directories) == set(expected)
        finally:
            Path(config_file).unlink()
    
    def test_str_method(self):
        config_data = {"key1": "value1", "key2": "value2"}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            assert str(config_manager) == str(config_data)
        finally:
            Path(config_file).unlink()
    
    def test_to_dict(self):
        config_data = {"key1": "value1", "key2": "value2"}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigManager(config_file)
            result = config_manager.to_dict()
            assert result == config_data
        finally:
            Path(config_file).unlink()
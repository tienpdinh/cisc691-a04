"""Tests for ConfigManager class."""
import pytest
import json
from pathlib import Path
from classes.config_manager import ConfigManager

class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_init_with_valid_config(self, config_file):
        """Test initialization with valid config file."""
        config = ConfigManager(config_file)
        assert config.config_file == Path(config_file)
        assert isinstance(config.settings, dict)
    
    def test_init_with_missing_config(self, temp_dir):
        """Test initialization with missing config file."""
        missing_config = temp_dir / "missing.json"
        with pytest.raises(SystemExit):
            ConfigManager(missing_config)
    
    def test_get_existing_key(self, config_file):
        """Test getting existing configuration key."""
        config = ConfigManager(config_file)
        assert config.get("log_level") == "INFO"
    
    def test_get_missing_key_with_default(self, config_file):
        """Test getting missing key with default value."""
        config = ConfigManager(config_file)
        assert config.get("missing_key", "default") == "default"
    
    def test_get_missing_key_without_default(self, config_file):
        """Test getting missing key without default."""
        config = ConfigManager(config_file)
        assert config.get("missing_key") is None
    
    def test_get_directory_names(self, config_file):
        """Test getting directory names from config."""
        config = ConfigManager(config_file)
        directories = config.get_directory_names()
        expected = [
            "data/raw_input",
            "data/cleaned_text", 
            "data/embeddings",
            "data/vectordb"
        ]
        assert set(directories) == set(expected)
    
    def test_to_dict(self, config_file, config_dict):
        """Test converting config to dictionary."""
        config = ConfigManager(config_file)
        assert config.to_dict() == config_dict
    
    def test_str_representation(self, config_file, config_dict):
        """Test string representation of config."""
        config = ConfigManager(config_file)
        assert str(config) == str(config_dict)
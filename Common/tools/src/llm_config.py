"""
LLM Configuration management.
Handles loading and managing LLM provider configurations.
"""

from typing import Dict, Any, Optional
import os
import yaml


class LLMServiceConfig:
    """
    Configuration manager for LLM providers.
    Supports loading configuration from a YAML configuration file
    """

    DEFAULT_CONFIG = {
        "default_provider": "ollama",
        "groq": {
            "model": "llama3-70b-8192",
            "api_key": None
        },
        "ollama": {
            "model": "llama3",
            "base_url": "http://localhost:11434"
        }
    }

    def __init__( self, config_file: Optional[str] = None ):
        """
        Initialize LLM configuration.

        Args:
            config_file: Optional path to YAML configuration file
        """
        if config_file and os.path.exists( config_file ):
            self._load_from_file( config_file )
        else:
            self.config = self.DEFAULT_CONFIG.copy()


    def _load_from_file( self, config_file: str ):
        """
        Load configuration from YAML file.

        Args:
            config_file: Path to configuration file
        """
        try:
            with open( config_file, 'r' ) as f:
                file_config = yaml.safe_load( f )

            # Merge loaded config with defaults
            if file_config:
                self.config = self.DEFAULT_CONFIG.copy()
                self._merge_config( self.config, file_config )
            else:
                self.config = self.DEFAULT_CONFIG.copy()
        except Exception as e:
            print( f"Warning: Failed to load LLM config from {config_file}: {e}" )
            self.config = self.DEFAULT_CONFIG.copy()

    def _merge_config( self, base: Dict, update: Dict ):
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance( base[key], dict ) and isinstance( value, dict ):
                self._merge_config( base[key], value )
            else:
                base[key] = value

    def get_provider_config( self, provider_name: str ) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider_name: Name of the provider (groq, ollama)

        Returns:
            Configuration dictionary for the provider
        """
        return self.config.get( provider_name, {} )

    def get_default_provider( self ) -> str:
        """Get the default provider name."""
        return self.config.get( "default_provider", "ollama" )

    def set_provider_config( self, provider_name: str, config: Dict[str, Any] ):
        """
        Set configuration for a provider.

        Args:
            provider_name: Provider name
            config: Configuration dictionary
        """
        if provider_name not in self.config:
            self.config[provider_name] = {}
        self.config[provider_name].update( config )


    def save_to_file( self, config_file: str ):
        """
        Save current configuration to a YAML file.

        Args:
            config_file: Path to save configuration
        """
        try:
            with open( config_file, 'w' ) as f:
                yaml.safe_dump( self.config, f, default_flow_style=False, sort_keys=False )
        except Exception as e:
            print( f"Error: Failed to save config to {config_file}: {e}" )

    def print_config( self ):
        """Print current configuration (with API keys redacted)."""
        safe_config = self._redact_secrets( self.config )
        print( yaml.safe_dump( safe_config, default_flow_style=False, sort_keys=False ) )

    def _redact_secrets( self, config: Dict ) -> Dict:
        """Create a copy of config with API keys redacted."""
        safe = {}
        for key, value in config.items():
            if isinstance( value, dict ):
                safe[key] = self._redact_secrets( value )
            elif "api_key" in key.lower() or "token" in key.lower():
                safe[key] = "***REDACTED***" if value else None
            else:
                safe[key] = value
        return safe


# Example configuration file format (llm_config.yaml):
"""
default_provider: groq

groq:
  model: llama3-70b-8192
  api_key: your-groq-api-key-here  # Get from https://console.groq.com

ollama:
  model: llama3
  base_url: http://localhost:11434  # Local Ollama server
"""

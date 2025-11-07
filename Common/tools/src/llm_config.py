"""
LLM Configuration management.
Handles loading and managing LLM provider configurations.
"""

from typing import Dict, Any, Optional
import os
import json
from llm_service import LLMService
from llm_provider_groq import GroqProvider
from llm_provider_ollama import OllamaProvider


class LLMConfig:
    """
    Configuration manager for LLM providers.

    Supports loading configuration from:
    1. Environment variables
    2. Configuration file
    3. Direct configuration
    """

    DEFAULT_CONFIG = {
        "default_provider": "ollama",
        "groq": {
            "model": "llama3-70b-8192",
            "api_key": None  # Set via env var GROQ_API_KEY or config file
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
            config_file: Optional path to JSON configuration file
        """
        if config_file and os.path.exists( config_file ):
            self._load_from_file( config_file )
        else:
            self.config = self.DEFAULT_CONFIG.copy()


    def _load_from_file( self, config_file: str ):
        """
        Load configuration from JSON file.

        Args:
            config_file: Path to configuration file
        """
        try:
            with open( config_file, 'r' ) as f:
                file_config = json.load( f )

            # Merge with existing config
            if "llm" in file_config:
                self._merge_config( self.config, file_config["llm"] )
            else:
                self._merge_config( self.config, file_config )
        except Exception as e:
            print( f"Warning: Failed to load LLM config from {config_file}: {e}" )

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

    def create_service( self ) -> LLMService:
        """
        Create and configure an LLMService instance with registered providers.
        """
        service = LLMService()

        # Register Groq provider
        groq_config = self.get_provider_config( "groq" )
        if groq_config.get( "api_key" ):
            try:
                groq_provider = GroqProvider( groq_config )
                service.register_provider( "groq", groq_provider )
            except Exception as e:
                print( f"Warning: Failed to initialize Groq provider: {e}" )

        # Register Ollama provider
        ollama_config = self.get_provider_config( "ollama" )
        try:
            ollama_provider = OllamaProvider( ollama_config )
            service.register_provider( "ollama", ollama_provider )
        except Exception as e:
            print( f"Warning: Failed to initialize Ollama provider: {e}" )

        # Set default provider
        default_provider = self.get_default_provider()
        try:
            service.set_provider( default_provider )
        except ValueError:
            print( f"Warning: Default provider '{default_provider}' not available" )

        return service

    def save_to_file( self, config_file: str ):
        """
        Save current configuration to a JSON file.

        Args:
            config_file: Path to save configuration
        """
        try:
            with open( config_file, 'w' ) as f:
                json.dump( {"llm": self.config}, f, indent=2 )
        except Exception as e:
            print( f"Error: Failed to save config to {config_file}: {e}" )

    def print_config( self ):
        """Print current configuration (with API keys redacted)."""
        safe_config = self._redact_secrets( self.config )
        print( json.dumps( safe_config, indent=2 ) )

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


# Example configuration file format:
"""
{
  "llm": {
    "default_provider": "groq",
    "groq": {
      "model": "llama3-70b-8192",
      "api_key": "your-groq-api-key-here"
    },
    "ollama": {
      "model": "llama3",
      "base_url": "http://localhost:11434"
    }
  }
}
"""

"""
Base abstract class for LLM providers.
Defines the interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class LLMProviderBase( ABC ):
    """
    Abstract base class for LLM providers.

    All LLM provider implementations must inherit from this class
    and implement the required methods.
    """

    def __init__( self, config: Dict[str, Any] ):
        """
        Initialize the LLM provider with configuration.

        Args:
            config: Dictionary containing provider-specific configuration
                   (API keys, model names, endpoints, etc.)
        """
        self.config = config
        self._is_configured = False

    @abstractmethod
    def generate( self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.7 ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt/query
            system_prompt: Optional system prompt to set context/behavior
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Generated text response from the LLM

        Raises:
            Exception: If generation fails or provider is not configured
        """
        pass

    @abstractmethod
    def is_available( self ) -> bool:
        """
        Check if the LLM provider is available and properly configured.

        Returns:
            True if provider is available and can be used, False otherwise
        """
        pass

    @abstractmethod
    def get_provider_name( self ) -> str:
        """
        Get the name of the LLM provider.

        Returns:
            String name of the provider (e.g., "OpenAI", "Anthropic", "Ollama")
        """
        pass

    def configure( self, config: Dict[str, Any] ) -> None:
        """
        Update configuration for the provider.

        Args:
            config: New configuration dictionary
        """
        self.config.update( config )
        self._is_configured = True

    def get_model_info( self ) -> Dict[str, Any]:
        """
        Get information about the currently configured model.

        Returns:
            Dictionary with model information (name, parameters, etc.)
        """
        return {
            "provider": self.get_provider_name(),
            "model": self.config.get( "model", "unknown" ),
            "configured": self._is_configured
        }

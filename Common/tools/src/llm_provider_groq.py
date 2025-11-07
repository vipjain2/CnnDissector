"""
Groq provider implementation.
"""

from typing import Dict, Any, Optional
from llm_provider_base import LLMProviderBase


class GroqProvider( LLMProviderBase ):
    """
    Groq provider implementation.

    Configuration required:
        - api_key: Groq API key
        - model: Model name (default: "llama3-70b-8192")

    Available models:
        - llama3-70b-8192
        - llama3-8b-8192
        - mixtral-8x7b-32768
        - gemma-7b-it
    """

    def __init__( self, config: Dict[str, Any] ):
        super().__init__( config )
        self._client = None
        self._initialize_client()

    def _initialize_client( self ):
        """Initialize the Groq client if API key is available."""
        try:
            from groq import Groq

            api_key = self.config.get( "api_key" )
            if not api_key:
                return

            self._client = Groq( api_key=api_key )
            self._is_configured = True
        except ImportError:
            print( "Warning: groq package not installed. Run: pip install groq" )
        except Exception as e:
            print( f"Warning: Failed to initialize Groq client: {e}" )

    def generate( self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.7 ) -> str:
        """
        Generate a response using Groq.

        Args:
            prompt: The user prompt/query
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated text response

        Raises:
            Exception: If client is not initialized or API call fails
        """
        if not self._client:
            raise Exception( "Groq client not initialized. Check API key configuration." )

        messages = []
        if system_prompt:
            messages.append( {"role": "system", "content": system_prompt} )
        messages.append( {"role": "user", "content": prompt} )

        model = self.config.get( "model", "llama3-70b-8192" )

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens or 8192,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception( f"Groq API call failed: {e}" )

    def is_available( self ) -> bool:
        """
        Check if Groq provider is available.

        Returns:
            True if client is initialized and configured
        """
        return self._client is not None and self._is_configured

    def get_provider_name( self ) -> str:
        """Get provider name."""
        return "Groq"

    def configure( self, config: Dict[str, Any] ) -> None:
        """
        Update configuration and reinitialize client.

        Args:
            config: New configuration dictionary
        """
        super().configure( config )
        self._initialize_client()

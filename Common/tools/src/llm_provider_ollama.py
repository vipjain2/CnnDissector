"""
Ollama provider implementation for local LLM models.
"""

from typing import Dict, Any, Optional
from llm_provider_base import LLMProviderBase


class OllamaProvider( LLMProviderBase ):
    """
    Ollama provider implementation for running local LLM models.

    Configuration required:
        - model: Model name (e.g., "llama3", "mistral", "codellama")
        - base_url: Ollama server URL (default: "http://localhost:11434")

    Popular models:
        - llama3 (8B, 70B)
        - mistral
        - mixtral
        - codellama
        - gemma
        - phi3

    Note: Models must be pulled first using: ollama pull <model-name>
    """

    def __init__( self, config: Dict[str, Any] ):
        super().__init__( config )
        self._client = None
        self._initialize_client()

    def _initialize_client( self ):
        """Initialize the Ollama client."""
        try:
            import ollama

            # Ollama doesn't require explicit initialization with API keys
            # Just check if the package is available
            self._client = ollama
            self._is_configured = True
        except ImportError:
            print( "Warning: ollama package not installed. Run: pip install ollama" )
        except Exception as e:
            print( f"Warning: Failed to initialize Ollama client: {e}" )

    def generate( self, prompt: str, system_prompt: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: float = 0.7 ) -> str:
        """
        Generate a response using Ollama.

        Args:
            prompt: The user prompt/query
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens in response (note: not all models support this)
            temperature: Sampling temperature

        Returns:
            Generated text response

        Raises:
            Exception: If client is not initialized or API call fails
        """
        if not self._client:
            raise Exception( "Ollama client not initialized. Install ollama package." )

        messages = []
        if system_prompt:
            messages.append( {"role": "system", "content": system_prompt} )
        messages.append( {"role": "user", "content": prompt} )

        model = self.config.get( "model", "llama3" )
        base_url = self.config.get( "base_url" )

        try:
            # Build options dict
            options = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            # Call Ollama
            kwargs = {
                "model": model,
                "messages": messages,
                "options": options
            }

            if base_url:
                kwargs["host"] = base_url

            response = self._client.chat( **kwargs )
            return response["message"]["content"]
        except Exception as e:
            raise Exception( f"Ollama API call failed: {e}. "
                          f"Make sure Ollama is running and model '{model}' is pulled." )

    def is_available( self ) -> bool:
        """
        Check if Ollama provider is available.

        Returns:
            True if client is initialized and Ollama server is reachable
        """
        if not self._client or not self._is_configured:
            return False

        # Try to check if Ollama server is running
        try:
            model = self.config.get( "model", "llama3" )
            base_url = self.config.get( "base_url" )

            kwargs = {"model": model}
            if base_url:
                kwargs["host"] = base_url

            # Try to list models to verify connection
            self._client.list( **kwargs )
            return True
        except:
            return False

    def get_provider_name( self ) -> str:
        """Get provider name."""
        return "Ollama"

    def configure( self, config: Dict[str, Any] ) -> None:
        """
        Update configuration.

        Args:
            config: New configuration dictionary
        """
        super().configure( config )
        # No need to reinitialize for Ollama

    def list_models( self ):
        """
        List available Ollama models.

        Returns:
            List of available model names

        Raises:
            Exception: If client is not initialized
        """
        if not self._client:
            raise Exception( "Ollama client not initialized." )

        try:
            base_url = self.config.get( "base_url" )
            kwargs = {}
            if base_url:
                kwargs["host"] = base_url

            models = self._client.list( **kwargs )
            return [model["name"] for model in models["models"]]
        except Exception as e:
            raise Exception( f"Failed to list Ollama models: {e}" )

"""
LLM Service for neural network analysis and description generation.
Handles context formatting and provider management.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from llm_provider_base import LLMProviderBase
from llm_provider_groq import GroqProvider
from llm_provider_ollama import OllamaProvider
from llm_config import LLMServiceConfig

class LLMService:
    """
    Service class for LLM-based neural network analysis.

    This class provides high-level methods for generating descriptions
    of neural network layers, architectures, and providing optimization suggestions.
    """

    SYSTEM_PROMPT = """You are an expert in deep learning and neural network architectures.
Your role is to analyze neural network layers and provide:
1. Clear descriptions of what the layer does
2. Explanation of the layer's parameters and their significance
3. How the layer fits in the overall architecture
4. Practical suggestions for improving performance or efficiency

Be concise, technical, and actionable in your responses."""

    def __init__( self, config_file ):
        self._providers: Dict[str, LLMProviderBase] = {}
        try:
            self.config = LLMServiceConfig( config_file )
        except:
            print( f"WARNING: Failed to initialize LLM : {e}" )

    def create_service( self ):
        """
        Create and configure an LLMService instance with registered providers.
        """
        provider_name = self.config.get_default_provider()

        # Register default provider
        llm_config = self.config.get_provider_config( provider_name )
        try:
            # Dynamically create provider class from provider name
            # e.g., "groq" -> "GroqProvider"
            class_name = provider_name[0].upper() + provider_name[1:] + "Provider"
            provider_class = globals()[class_name]
            provider_obj = provider_class( llm_config )

            self.register_provider( provider_name, provider_obj )
            self.set_provider( provider_name )

        except Exception as e:
            print( f"WARNING: Failed to initialize LLM : {e}" )


    def register_provider( self, name: str, provider: LLMProviderBase ) -> None:
        """
        Register an LLM provider.

        Args:
            name: Provider name (e.g., "groq", "ollama")
            provider: Provider instance
        """
        self._providers[name] = provider


    def set_provider( self, name: str ) -> None:
        """
        Set the active LLM provider.

        Args:
            name: Name of registered provider

        Raises:
            ValueError: If provider not found
        """
        if name not in self._providers:
            raise ValueError( f"Provider '{name}' not registered. "
                           f"Available: {list( self._providers.keys() )}" )
        self._provider = self._providers[name]


    def get_provider( self ) -> Optional[LLMProviderBase]:
        """Get the currently active provider."""
        return self._provider


    def is_available( self ) -> bool:
        """Check if LLM service is available (has a configured provider)."""
        return self._provider is not None and self._provider.is_available()


    def _format_layer_info( self, layer: nn.Module, layer_name: str ) -> str:
        """
        Format layer information for LLM prompt.

        Args:
            layer: PyTorch layer module
            layer_name: Name/identifier of the layer
        """
        info = [f"Layer: {layer_name}", f"Type: {type( layer ).__name__}", ""]

        # Get layer parameters
        total_params = sum( p.numel() for p in layer.parameters() )
        trainable_params = sum( p.numel() for p in layer.parameters() if p.requires_grad )
        info.append( f"Total Parameters: {total_params:,}" )
        info.append( f"Trainable Parameters: {trainable_params:,}" )
        info.append( "" )

        # Layer-specific information
        if isinstance( layer, nn.Conv2d ):
            info.extend( [
                f"Input Channels: {layer.in_channels}",
                f"Output Channels: {layer.out_channels}",
                f"Kernel Size: {layer.kernel_size}",
                f"Stride: {layer.stride}",
                f"Padding: {layer.padding}",
                f"Dilation: {layer.dilation}",
                f"Groups: {layer.groups}",
                f"Bias: {layer.bias is not None}"
            ] )
        elif isinstance( layer, nn.Linear ):
            info.extend( [
                f"Input Features: {layer.in_features}",
                f"Output Features: {layer.out_features}",
                f"Bias: {layer.bias is not None}"
            ] )
        elif isinstance( layer, nn.BatchNorm2d ):
            info.extend( [
                f"Num Features: {layer.num_features}",
                f"Epsilon: {layer.eps}",
                f"Momentum: {layer.momentum}",
                f"Affine: {layer.affine}",
                f"Track Running Stats: {layer.track_running_stats}"
            ] )
        elif isinstance( layer, ( nn.MaxPool2d, nn.AvgPool2d ) ):
            info.extend( [
                f"Kernel Size: {layer.kernel_size}",
                f"Stride: {layer.stride}",
                f"Padding: {layer.padding}"
            ] )
        elif isinstance( layer, nn.ReLU ):
            info.append( f"Inplace: {layer.inplace}" )
        elif isinstance( layer, nn.Dropout ):
            info.extend( [
                f"Dropout Probability: {layer.p}",
                f"Inplace: {layer.inplace}"
            ] )

        return "\n".join( info )


    def _format_activation_stats( self, activations: torch.Tensor ) -> str:
        """
        Format activation statistics for LLM prompt.

        Args:
            activations: Tensor containing layer activations
        """
        if activations is None:
            return "Activations: Not available"

        info = ["Activation Statistics:"]
        info.append( f"Shape: {tuple( activations.shape )}" )
        info.append( f"Mean: {activations.mean().item():.6f}" )
        info.append( f"Std: {activations.std().item():.6f}" )
        info.append( f"Min: {activations.min().item():.6f}" )
        info.append( f"Max: {activations.max().item():.6f}" )
        info.append( f"Sparsity: {( activations == 0 ).float().mean().item():.2%}" )

        return "\n".join( info )


    def _format_weight_stats( self, layer: nn.Module ) -> str:
        """
        Format weight statistics for LLM prompt.

        Args:
            layer: PyTorch layer module
        """
        info = ["Weight Statistics:"]

        if hasattr( layer, 'weight' ) and layer.weight is not None:
            weight = layer.weight.data
            info.append( f"Weight Shape: {tuple( weight.shape )}" )
            info.append( f"Weight Mean: {weight.mean().item():.6f}" )
            info.append( f"Weight Std: {weight.std().item():.6f}" )
            info.append( f"Weight Min: {weight.min().item():.6f}" )
            info.append( f"Weight Max: {weight.max().item():.6f}" )

        if hasattr( layer, 'bias' ) and layer.bias is not None:
            bias = layer.bias.data
            info.append( f"Bias Shape: {tuple( bias.shape )}" )
            info.append( f"Bias Mean: {bias.mean().item():.6f}" )
            info.append( f"Bias Std: {bias.std().item():.6f}" )

        return "\n".join( info ) if len( info ) > 1 else "Weights: Not available"


    def describe_layer( self, layer: nn.Module, layer_name: str,
                      activations: Optional[torch.Tensor] = None,
                      architecture_context: Optional[str] = None,
                      include_suggestions: bool = True ) -> str:
        """
        Generate a comprehensive description of a neural network layer.

        Args:
            layer: PyTorch layer module
            layer_name: Name/identifier of the layer
            activations: Optional tensor with layer activations
            architecture_context: Optional context about the overall network architecture
            include_suggestions: Whether to include optimization suggestions
        """
        if not self.is_available():
            raise Exception( "LLM service not available. Configure a provider first." )

        # Build the prompt
        prompt_parts = ["Please analyze the following neural network layer:\n"]

        # Layer information
        prompt_parts.append( self._format_layer_info( layer, layer_name ) )
        prompt_parts.append( "" )

        # Weight statistics
        prompt_parts.append( self._format_weight_stats( layer ) )
        prompt_parts.append( "" )

        # Activation statistics
        if activations is not None:
            prompt_parts.append( self._format_activation_stats( activations ) )
            prompt_parts.append( "" )

        # Architecture context
        if architecture_context:
            prompt_parts.append( f"Architecture Context:\n{architecture_context}" )
            prompt_parts.append( "" )

        # Request
        prompt_parts.append( "Please provide:" )
        prompt_parts.append( "1. A description of what this layer does" )
        prompt_parts.append( "2. The significance of its parameters" )
        prompt_parts.append( "3. How it contributes to the network" )

        if include_suggestions:
            prompt_parts.append( "4. Suggestions for improving performance or efficiency" )

        prompt = "\n".join( prompt_parts )

        return self._provider.generate( prompt, system_prompt=self.SYSTEM_PROMPT )


    def describe_architecture( self, model: nn.Module, model_name: str = None ) -> str:
        """
        Generate a high-level description of the entire network architecture.
        """
        if not self.is_available():
            raise Exception( "LLM service not available. Configure a provider first." )

        # Build model summary
        total_params = sum( p.numel() for p in model.parameters() )
        trainable_params = sum( p.numel() for p in model.parameters() if p.requires_grad )

        prompt_parts = [
            f"Please analyze the following neural network architecture:\n",
            f"Model: {model_name}",
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            "",
            "Architecture:",
            str( model ),
            "",
            "Please provide:",
            "1. An overview of the architecture type and design",
            "2. Analysis of the layer composition and flow",
            "3. Potential use cases for this architecture",
            "4. Suggestions for architecture improvements"
        ]

        prompt = "\n".join( prompt_parts )

        return self._provider.generate( prompt, system_prompt=self.SYSTEM_PROMPT,
                                      max_tokens=2000 )

    def suggest_optimizations( self, model: nn.Module,
                            training_stats: Optional[Dict[str, Any]] = None ) -> str:
        """
        Generate optimization suggestions for the model.

        Args:
            model: PyTorch model
            training_stats: Optional dictionary with training statistics
                          (loss, accuracy, convergence info, etc.)
        """
        if not self.is_available():
            raise Exception( "LLM service not available. Configure a provider first." )

        total_params = sum( p.numel() for p in model.parameters() )

        prompt_parts = [
            "Please suggest optimizations for the following neural network:\n",
            f"Total Parameters: {total_params:,}",
            "",
            "Architecture:",
            str( model ),
            ""
        ]

        if training_stats:
            prompt_parts.append( "Training Statistics:" )
            for key, value in training_stats.items():
                prompt_parts.append( f"{key}: {value}" )
            prompt_parts.append( "" )

        prompt_parts.extend( [
            "Please provide specific suggestions for:",
            "1. Improving model performance (accuracy/loss)",
            "2. Reducing computational cost and memory usage",
            "3. Speeding up training convergence",
            "4. Architecture modifications to consider"
        ] )

        prompt = "\n".join( prompt_parts )

        return self._provider.generate( prompt, system_prompt=self.SYSTEM_PROMPT,
                                      max_tokens=2000 )

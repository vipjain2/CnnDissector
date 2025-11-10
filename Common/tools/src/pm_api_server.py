#!/usr/bin/env python3
"""
Stateful FastAPI server for PyTorch Model Shell.

Each user runs their own instance of this server, maintaining all state on the server side.
The frontend communicates via REST API endpoints.
"""

import os
import sys
from typing import Optional, Dict, Any, List
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr, asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import pmshell - need to handle imports carefully
import cmd


# Pydantic models for request/response
class CommandRequest( BaseModel ):
    command: str


class CommandResponse( BaseModel ):
    output: str
    success: bool
    error: Optional[str] = None


class ConfigGetResponse( BaseModel ):
    config: Dict[str, Any]


class ConfigUpdateRequest( BaseModel ):
    key: str
    value: Any


class ModelInfoResponse( BaseModel ):
    has_model: bool
    model_info: Optional[str] = None


# Global shell instance - maintains all state
shell_instance = None


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan( app: FastAPI ):
    """Manage application lifespan (startup and shutdown)."""
    global shell_instance

    # Startup
    print( "PyTorch Model Shell API server starting..." )

    # Import pmshell module by loading it dynamically
    import importlib.util
    import importlib.machinery
    script_dir = os.path.dirname( os.path.abspath( __file__ ) )
    pmshell_path = os.path.join( script_dir, "pmshell" )

    # Load pmshell as a module using SourceFileLoader (works with files without .py extension)
    loader = importlib.machinery.SourceFileLoader( "pmshell_module", pmshell_path )
    spec = importlib.util.spec_from_loader( loader.name, loader )
    pmshell_module = importlib.util.module_from_spec( spec )
    sys.modules["pmshell_module"] = pmshell_module
    loader.exec_module( pmshell_module )

    # Get Shell and Config classes
    Shell = pmshell_module.Shell
    Config = pmshell_module.Config

    config = Config()
    shell_instance = Shell( config )

    print( "PyTorch Model Shell API server started" )
    print( f"Shell instance initialized with config" )

    yield  # Server runs here

    # Shutdown
    print( "PyTorch Model Shell API server shutting down" )


# FastAPI app with lifespan
app = FastAPI(
    title="PyTorch Model Shell API",
    description="Stateful REST API for pmshell - one instance per user",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def capture_output( func, *args, **kwargs ):
    """
    Capture stdout and stderr from a function call.

    Returns:
        Tuple of (stdout_content, stderr_content, exception_if_any)
    """
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    exception = None

    try:
        with redirect_stdout( stdout_buffer ), redirect_stderr( stderr_buffer ):
            func( *args, **kwargs )
    except Exception as e:
        exception = e

    return stdout_buffer.getvalue(), stderr_buffer.getvalue(), exception


# API Endpoints

@app.get( "/health" )
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "has_shell": shell_instance is not None
    }


@app.get( "/server/pid" )
async def get_server_pid():
    """Get the server process ID."""
    return {
        "pid": os.getpid()
    }


@app.post( "/server/shutdown" )
async def shutdown_server():
    """Shutdown the server gracefully."""
    import signal

    # Schedule shutdown after responding
    def shutdown():
        os.kill( os.getpid(), signal.SIGTERM )

    # Import threading to schedule shutdown
    import threading
    threading.Timer( 0.5, shutdown ).start()

    return {"message": "Server shutting down"}


@app.post( "/command", response_model=CommandResponse )
async def execute_command( request: CommandRequest ):
    """
    Execute a pmshell command.

    The command is executed in the context of the global shell instance,
    maintaining all state between calls.
    """
    if shell_instance is None:
        raise HTTPException( status_code=500, detail="Shell not initialized" )

    # Set up API output buffer
    output_buffer = StringIO()
    shell_instance.api_output = output_buffer

    exception = None
    try:
        shell_instance.onecmd( shell_instance.precmd( request.command ) )
    except Exception as e:
        exception = e
    finally:
        # Clear API output buffer
        shell_instance.api_output = None

    # Get captured output
    full_output = output_buffer.getvalue()

    if exception:
        return CommandResponse(
            output=full_output,
            success=False,
            error=str( exception )
        )

    return CommandResponse(
        output=full_output.strip(),
        success=True
    )


@app.get( "/config", response_model=ConfigGetResponse )
async def get_config():
    """Get current configuration."""
    if shell_instance is None:
        raise HTTPException( status_code=500, detail="Shell not initialized" )

    config = shell_instance.config

    # Convert config to dict
    config_dict = {}
    for attr in dir( config ):
        if not attr.startswith( "_" ) and not callable( getattr( config, attr ) ):
            value = getattr( config, attr )
            # Only include serializable types
            if isinstance( value, ( str, int, float, bool, list, dict, type( None ) ) ):
                config_dict[attr] = value

    return ConfigGetResponse( config=config_dict )


@app.post( "/config" )
async def update_config( request: ConfigUpdateRequest ):
    """Update a configuration value."""
    if shell_instance is None:
        raise HTTPException( status_code=500, detail="Shell not initialized" )

    config = shell_instance.config

    if not hasattr( config, request.key ):
        raise HTTPException(
            status_code=400,
            detail=f"Configuration key '{request.key}' does not exist"
        )

    setattr( config, request.key, request.value )

    return {"message": f"Configuration '{request.key}' updated successfully"}


@app.get( "/model/info", response_model=ModelInfoResponse )
async def get_model_info():
    """Get information about the currently loaded model."""
    if shell_instance is None:
        raise HTTPException( status_code=500, detail="Shell not initialized" )

    # Access global model variable from pmshell module
    pmshell_module = sys.modules.get( "pmshell_module" )

    has_model = False
    model_info = None

    if pmshell_module and hasattr( pmshell_module, "model" ):
        has_model = pmshell_module.model is not None
        if has_model:
            model_info = str( type( pmshell_module.model ).__name__ )

    return ModelInfoResponse(
        has_model=has_model,
        model_info=model_info
    )


@app.post( "/model/summary" )
async def get_model_summary():
    """Get model summary by executing the summary command."""
    if shell_instance is None:
        raise HTTPException( status_code=500, detail="Shell not initialized" )

    stdout_output, stderr_output, exception = capture_output(
        shell_instance.do_summary,
        ""
    )

    if exception:
        raise HTTPException( status_code=500, detail=str( exception ) )

    return {"summary": stdout_output.strip()}


@app.get( "/llm/status" )
async def get_llm_status():
    """Get LLM service status."""
    if shell_instance is None:
        raise HTTPException( status_code=500, detail="Shell not initialized" )

    has_llm = shell_instance.llm_service is not None
    is_available = False
    provider_name = None

    if has_llm:
        is_available = shell_instance.llm_service.is_available()
        if is_available:
            provider = shell_instance.llm_service.get_provider()
            provider_name = provider.get_provider_name()

    return {
        "has_llm": has_llm,
        "is_available": is_available,
        "provider": provider_name
    }


# Main entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser( description="PyTorch Model Shell API Server" )
    parser.add_argument( "--host", default="127.0.0.1", help="Host to bind to" )
    parser.add_argument( "--port", type=int, default=8000, help="Port to bind to" )
    parser.add_argument( "--reload", action="store_true", help="Enable auto-reload for development" )

    args = parser.parse_args()

    print( f"Starting PyTorch Model Shell API server on {args.host}:{args.port}" )

    uvicorn.run(
        "pm_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

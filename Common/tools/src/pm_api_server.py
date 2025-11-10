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


# Global shell instance - maintains all state
shell_instance = None
model = None
image = None


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan( app: FastAPI ):
    """Manage application lifespan (startup and shutdown)."""
    global shell_instance, model, image

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
    shell_instance = Shell( config, server_mode=True )

    print( "PyTorch Model Shell API server started" )

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


@app.get( "/server/output" )
async def get_server_output():
    """Get and clear the server output buffer."""
    if shell_instance is None:
        raise HTTPException( status_code=500, detail="Shell not initialized" )

    output = shell_instance.get_output()
    return {"output": output}


@app.post( "/command", response_model=CommandResponse )
async def execute_command( request: CommandRequest ):
    """
    Execute a pmshell command.

    The command is executed in the context of the global shell instance,
    maintaining all state between calls.
    """
    if shell_instance is None:
        raise HTTPException( status_code=500, detail="Shell not initialized" )

    exception = None
    try:
        shell_instance.onecmd( shell_instance.precmd( request.command ) )
    except Exception as e:
        exception = e

    # Get captured output from server mode buffer
    full_output = shell_instance.get_output()

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

#!/usr/bin/env python3
"""
Stateful FastAPI server for PyTorch Model Shell.

Each user runs their own instance of this server, maintaining all state on the server side.
The frontend communicates via REST API endpoints.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr, asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cmd

# Configure logging for server lifecycle events
# Get log file path from environment variable or default to server code directory
script_dir = os.path.dirname(os.path.abspath(__file__))
default_log_path = os.path.join(script_dir, 'server.log')
log_file = os.getenv('PMSHELL_SERVER_LOG_PATH', default_log_path)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [PID:%(process)d] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file)  # Only log to file, not to console
    ]
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class CommandRequest( BaseModel ):
    command: str

class CommandResponse( BaseModel ):
    output: str
    success: bool
    error: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded PNG image data


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
    logger.info( "PyTorch Model Shell API server starting..." )

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

    # Get Shell, Config, and prepare_shell function
    Shell = pmshell_module.Shell
    Config = pmshell_module.Config
    prepare_shell = pmshell_module.prepare_shell

    # Configure matplotlib backend before creating shell
    prepare_shell( server_mode=True )

    config = Config()
    shell_instance = Shell( config, server_mode=True )

    logger.info( "PyTorch Model Shell API server started" )

    yield  # Server runs here

    # Shutdown
    logger.info( "PyTorch Model Shell API server shutting down" )


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

    # Log the command being executed
    logger.info( f"Requested: {request.command}" )

    exception = None
    try:
        shell_instance.onecmd( shell_instance.precmd( request.command ) )
    except Exception as e:
        exception = e

    # Get captured output from server mode buffer
    full_output = shell_instance.get_output()

    # Check if there's an image to return
    image_data = shell_instance.fig.get_image_data()
    if image_data:
        # Clear the buffer after retrieving
        shell_instance.fig.clear_image_buffer()

    if exception:
        return CommandResponse(
            output=full_output,
            success=False,
            error=str( exception ),
            image_data=image_data
        )

    return CommandResponse(
        output=full_output.strip(),
        success=True,
        image_data=image_data
    )


# Main entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser( description="PyTorch Model Shell API Server" )
    parser.add_argument( "--host", default="127.0.0.1", help="Host to bind to" )
    parser.add_argument( "--port", type=int, default=8000, help="Port to bind to" )
    parser.add_argument( "--reload", action="store_true", help="Enable auto-reload for development" )

    args = parser.parse_args()

    logger.info( f"Starting PyTorch Model Shell API server on {args.host}:{args.port}" )

    # Configure uvicorn to log to file only (not console)
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "[%(asctime)s] [PID:%(process)d] %(levelprefix)s %(message)s"
    log_config["formatters"]["access"]["fmt"] = '[%(asctime)s] [PID:%(process)d] %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'

    # Replace console handlers with file handlers for all loggers
    log_config["handlers"]["default"] = {
        "class": "logging.FileHandler",
        "filename": log_file,
        "formatter": "default"
    }
    log_config["handlers"]["access"] = {
        "class": "logging.FileHandler",
        "filename": log_file,
        "formatter": "access"
    }

    uvicorn.run(
        "pm_api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=log_config
    )

# PMShell Frontend

Terminal-based frontend for PyTorch Model Shell with rich formatting.

## Features

- Clean command-line interface
- Syntax highlighting and colored output
- Formatted tables and sections
- Error and warning highlighting

## Installation

```bash
npm install
```

## Usage

```bash
npm start
```

Or make it executable and run directly:

```bash
chmod +x index.js
./index.js
```

## Commands

All pmshell commands are supported. Examples:

- `summary` - Show model summary
- `show config` - Display configuration
- `nparams` - Show parameter count
- `quit` or `exit` - Exit the shell

## Output Formatting

The frontend provides enhanced formatting:

- **Cyan**: Headers and model names
- **Yellow**: Values and configuration settings
- **Red**: Errors
- **Yellow with ⚠**: Warnings
- **Magenta**: LLM provider information
- **Gray**: Keys and metadata
- **Dim**: Separator lines

## Requirements

- Node.js 12+ (tested with v18.19.1)
- Python 3 with pmshell backend


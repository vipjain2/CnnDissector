#!/usr/bin/env node

const { spawn, exec } = require('child_process');
const { createInterface } = require('readline');
const chalk = require('chalk');
const boxen = require('boxen');
const http = require('http');
const fs = require('fs');
const { marked } = require('marked');
const TerminalRenderer = require('marked-terminal');

class PMShellAPIFrontend {
  constructor() {
    this.apiServer = null;
    this.rl = null;
    this.apiHost = '127.0.0.1';
    this.apiPort = 8000;
    this.apiBaseUrl = `http://${this.apiHost}:${this.apiPort}`;
    this.isLogging = false;  // Flag to prevent recursive logging
    this.imageViewerUrl = 'http://127.0.0.1:3001';
    this.statusBarEnabled = false;
    this.originalConsoleLog = console.log.bind(console);
    this.isUpdatingStatusBar = false;  // Flag to prevent recursive status bar updates

    // Configure marked for terminal rendering
    marked.setOptions({
      renderer: new TerminalRenderer()
    });
  }

  async start() {
    console.log(boxen(
      chalk.cyan.bold('PyTorch Model Shell'),
      {
        padding: 1,
        margin: 1,
        borderStyle: 'round',
        borderColor: 'cyan'
      }
    ));

    // Check if server is already running
    const serverRunning = await this.checkServerRunning();

    if (serverRunning) {
      console.log(chalk.green('✓ Using existing backend server\n'));
    } else {
      console.log(chalk.gray('Starting API backend server...'));

      // Start the API server
      await this.startAPIServer();

      // Wait for server to be ready
      await this.waitForServer();

      console.log(chalk.green('✓ Backend server ready\n'));
    }

    // Setup status bar and console wrapper
    this.setupStatusBar();

    // Setup readline interface
    this.setupReadline();
  }

  async checkServerRunning() {
    try {
      await this.makeRequest('/health');
      return true;
    } catch (err) {
      return false;
    }
  }

  async getServerPid() {
    try {
      const response = await this.makeRequest('/server/pid');
      return response.pid;
    } catch (err) {
      return null;
    }
  }

  async logClientError(data) {
    // Prevent recursive logging
    if (this.isLogging) {
      return;
    }

    this.isLogging = true;

    try {
      const lines = data.split('\n');
      const pid = await this.getServerPid();
      const timestamp = new Date().toISOString();

      for (const line of lines) {
        const trimmedLine = line.trim();
        if (trimmedLine) {
          const logEntry = `[${timestamp}] [Server PID:${pid || 'unknown'}] ${trimmedLine}\n`;
          fs.appendFileSync('client.log', logEntry);
        }
      }
    } finally {
      this.isLogging = false;
    }
  }

  startAPIServer() {
    const serverPath = '../src/pm_api_server.py';

    return new Promise((resolve, reject) => {
      // Run the API server as detached process
      // Use the shebang to pick the correct Python from environment
      this.apiServer = spawn(serverPath, [
        '--host', this.apiHost,
        '--port', this.apiPort.toString()
      ], {
        env: process.env,
        detached: true,  // Run independently
        stdio: ['ignore', 'pipe', 'pipe']  // Pipe stdout and stderr for monitoring
      });

      // Server lifecycle logs are written to server.log by the server itself
      // We only log stderr (errors/crashes) on the client side
      this.apiServer.stdout.on('data', (data) => {
        // Display server startup messages on terminal for user feedback
        const output = data.toString().trim();
        if (output) {
          console.log(chalk.gray(output));
        }
      });

      this.apiServer.stderr.on('data', (data) => {
        // Log errors to client.log for debugging
        this.logClientError(data.toString());
      });

      // Detach the process so it can run independently
      this.apiServer.unref();

      // Handle process exit - don't crash client
      this.apiServer.on('close', (code) => {
        if (code !== 0) {
          console.log(chalk.yellow(`\n⚠ Backend server exited with code ${code}`));
        }
        // Don't call process.exit() - let client continue
      });

      this.apiServer.on('error', (err) => {
        console.error(chalk.red(`✗ Failed to start backend server: ${err.message}`));
        // Don't call process.exit() - let client continue
      });

      // Give it a moment to start
      setTimeout(resolve, 2000);
    });
  }

  async waitForServer(maxRetries = 10) {
    for (let i = 0; i < maxRetries; i++) {
      try {
        await this.makeRequest('/health');
        return;
      } catch (err) {
        if (i < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, 500));
        } else {
          throw new Error('Server failed to start');
        }
      }
    }
  }

  setupReadline() {
    this.rl = createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: chalk.green('pmshell> ')
    });

    // Wrap readline's output write to redraw status bar after any terminal write
    const originalWrite = process.stdout.write.bind(process.stdout);
    process.stdout.write = (...args) => {
      const result = originalWrite(...args);
      // Only update status bar if we're not already updating it (prevent recursion)
      if (!this.isUpdatingStatusBar) {
        this.updateStatusBar();
      }
      return result;
    };

    this.rl.prompt();

    this.rl.on('line', async (line) => {
      const input = line.trim();
      // Normalize input: replace spaces with underscores for command matching
      const normalizedInput = input.replace(/\s+/g, '_');

      if (input === 'quit' || input === 'exit') {
        this.rl.close();
        return;
      }

      if (input === 'help' || input === '?') {
        await this.showHelp();
        this.rl.prompt();
        return;
      }

      if (normalizedInput === 'kill_server') {
        try {
          await this.killServer();
        } catch (err) {
          console.log(chalk.red('✗ Failed to kill server: ' + err.message));
        }
        this.rl.prompt();
        return;
      }

      if (normalizedInput === 'restart_server') {
        try {
          await this.restartServer();
        } catch (err) {
          console.log(chalk.red('✗ Failed to restart server: ' + err.message));
        }
        this.rl.prompt();
        return;
      }

      if (input) {
        await this.sendCommand(input);
      } else {
        this.rl.prompt();
      }
    });

    this.rl.on('close', async () => {
      await this.shutdown();
    });
  }

  setupStatusBar() {
    this.statusBarEnabled = true;

    // Set scrolling region to exclude the last line
    // This tells the terminal to only scroll lines 1 to (rows-1)
    const scrollRegion = process.stdout.rows - 1;
    process.stdout.write(`\x1b[1;${scrollRegion}r`);

    // Move cursor to just above status bar (line rows-1)
    process.stdout.write(`\x1b[${scrollRegion};1H`);

    // Wrap console.log to redraw status after every output
    console.log = (...args) => {
      this.originalConsoleLog(...args);
      this.updateStatusBar();
    };

    // Handle terminal resize
    process.stdout.on('resize', () => {
      // Reset scroll region on resize
      const newScrollRegion = process.stdout.rows - 1;
      process.stdout.write(`\x1b[1;${newScrollRegion}r`);
      this.updateStatusBar();
    });

    // Initial render
    this.updateStatusBar();
  }

  updateStatusBar() {
    if (!this.statusBarEnabled) return;

    this.isUpdatingStatusBar = true;

    const termWidth = process.stdout.columns || 80;
    const rightText = 'NetDissect  ';
    const padding = ' '.repeat(Math.max(0, termWidth - rightText.length));
    const statusContent = padding + rightText;

    // Save cursor, move to last line (absolute), draw status, restore cursor
    process.stdout.write(
      '\x1b[s' +                                    // Save cursor position
      `\x1b[${process.stdout.rows};1H` +           // Move to last line (absolute)
      '\x1b[K' +                                    // Clear line
      '\x1b[44m\x1b[37m' +                          // Blue background, white text
      statusContent +
      '\x1b[0m' +                                   // Reset colors
      '\x1b[u'                                      // Restore cursor position
    );

    this.isUpdatingStatusBar = false;
  }

  async sendCommand(command) {
    try {
      const response = await this.makeRequest('/command', {
        method: 'POST',
        body: JSON.stringify({ command: command })
      });

      if (response.success) {
        if (response.output) {
          this.handleOutput(response.output);
        }
        if (response.image_data) {
          await this.displayImage(response.image_data);
        }
      } else {
        console.log(chalk.red('Error: ' + (response.error || 'Command failed')));
        if (response.image_data) {
          await this.displayImage(response.image_data);
        }
      }
    } catch (err) {
      console.log(chalk.red('✗ Request failed: ' + err.message));
    }

    this.rl.prompt();
  }

  isMarkdown(text) {
    // Only treat as markdown if it has clear structural markdown elements
    return /^#{1,6}\s/m.test(text) ||  // Headers
           /^```/m.test(text);          // Code blocks
  }

  async displayImage(imageData) {
    try {
      // Send image data to Next.js image viewer
      await this.makeRequest(`${this.imageViewerUrl}/api/image`, {
        method: 'POST',
        body: JSON.stringify({ imageData: imageData })
      });

      console.log(chalk.cyan('📊 Image sent to viewer at ' + this.imageViewerUrl));
    } catch (err) {
      console.log(chalk.yellow(`⚠ Could not display image: ${err.message}`));
      console.log(chalk.gray('   Make sure the image viewer is running: cd image-viewer && npm run dev'));
    }
  }

  handleOutput(output) {
    // Check if the entire output looks like markdown
    if (this.isMarkdown(output)) {
      try {
        const rendered = marked(output);
        console.log(rendered);
        return;
      } catch (err) {
        // If markdown rendering fails, fall through to line-by-line processing
      }
    }

    // Line-by-line processing for non-markdown output
    const lines = output.split('\n');

    lines.forEach(line => {
      if (line) {
        // Format different types of output
        if (line.startsWith('***')) {
          // Error messages
          console.log(chalk.red(line));
        } else if (line.match(/^-{3,}/)) {
          // Separator lines
          console.log(chalk.dim(line));
        } else {
          // Regular output
          console.log(line);
        }
      }
    });
  }

  makeRequest(endpoint, options = {}) {
    return new Promise((resolve, reject) => {
      const url = endpoint.startsWith('http') ? endpoint : this.apiBaseUrl + endpoint;
      const parsedUrl = new URL(url);

      const reqOptions = {
        hostname: parsedUrl.hostname,
        port: parsedUrl.port,
        path: parsedUrl.pathname,
        method: options.method || 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      };

      const req = http.request(reqOptions, (res) => {
        let data = '';

        res.on('data', (chunk) => {
          data += chunk;
        });

        res.on('end', () => {
          try {
            const json = JSON.parse(data);
            if (res.statusCode >= 200 && res.statusCode < 300) {
              resolve(json);
            } else {
              reject(new Error(json.detail || `HTTP ${res.statusCode}`));
            }
          } catch (err) {
            reject(new Error('Invalid JSON response'));
          }
        });
      });

      req.on('error', (err) => {
        reject(err);
      });

      if (options.body) {
        req.write(options.body);
      }

      req.end();
    });
  }

  async safeRequest(endpoint, options = {}) {
    try {
      return await this.makeRequest(endpoint, options);
    } catch (err) {
      return null;
    }
  }

  async showHelp() {
    console.log(chalk.cyan.bold('\nPMShell Client Commands:'));
    console.log(chalk.gray('─'.repeat(50)));

    console.log(chalk.green('  help, ?') + '               ' + chalk.white('Show this help message'));
    console.log(chalk.green('  kill server') + '           ' + chalk.white('Terminate the backend API server'));
    console.log(chalk.green('  restart server') + '        ' + chalk.white('Restart the backend API server'));
    console.log(chalk.green('  quit, exit') + '            ' + chalk.white('Exit the client'));

    // Get pmshell help from backend
    const response = await this.safeRequest('/command', {
      method: 'POST',
      body: JSON.stringify({ command: 'help' })
    });

    if (response && response.success && response.output) {
      console.log(chalk.cyan.bold('\nPMShell Server Commands:'));
      console.log(chalk.gray('─'.repeat(50)));
      console.log(response.output);
    } else {
      console.log(chalk.yellow('\n⚠ Could not fetch pmshell help from backend'));
      console.log(chalk.gray('Type pmshell commands directly to execute them on the backend.\n'));
    }
  }

  async killServer() {
    console.log(chalk.gray('Sending shutdown command to server...'));

    try {
      // Send shutdown request to server
      await this.makeRequest('/server/shutdown', { method: 'POST' });
      console.log(chalk.green('✓ Shutdown command sent'));

      // Wait a moment for server to shut down
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Verify server has exited
      const serverRunning = await this.checkServerRunning();

      if (serverRunning) {
        console.log(chalk.yellow('⚠ Server is still running'));
      } else {
        console.log(chalk.green('✓ Server has shut down'));
      }
    } catch (err) {
      // If request fails, server might already be down
      const serverRunning = await this.checkServerRunning();

      if (!serverRunning) {
        console.log(chalk.yellow('⚠ Server is not running'));
      } else {
        console.log(chalk.red('✗ Failed to shutdown server: ' + err.message));
      }
    }
  }

  async restartServer() {
    console.log(chalk.cyan('Restarting server...\n'));

    // Kill the existing server (already handles verification)
    await this.killServer();

    // Start a new server
    console.log(chalk.gray('\nStarting new server...'));
    await this.startAPIServer();
    await this.waitForServer();

    // Fetch and display server startup messages
    const response = await this.safeRequest('/server/output');
    if (response && response.output) {
      console.log(response.output);
    }

    console.log(chalk.green('✓ Server restarted successfully\n'));
  }

  async shutdown() {
    // Restore console.log
    if (this.originalConsoleLog) {
      console.log = this.originalConsoleLog;
    }

    // Reset scroll region to full screen and clear status bar
    if (this.statusBarEnabled) {
      process.stdout.write('\x1b[r');                 // Reset scroll region
      const termWidth = process.stdout.columns || 80;
      process.stdout.write(
        `\x1b[${process.stdout.rows};1H` +           // Move to last line
        '\x1b[K' +                                    // Clear line
        '\n'                                          // Newline
      );
    }

    console.log(chalk.cyan('Shutting down client...'));

    if (this.rl) {
      this.rl.close();
    }

    process.exit(0);
  }
}

// Start the frontend
const frontend = new PMShellAPIFrontend();
frontend.start().catch(err => {
  console.error(chalk.red('Failed to start:', err.message));
  process.exit(1);
});

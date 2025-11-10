#!/usr/bin/env node

const { spawn } = require('child_process');
const { createInterface } = require('readline');
const chalk = require('chalk');
const boxen = require('boxen');
const http = require('http');
const fs = require('fs');

class PMShellAPIFrontend {
  constructor() {
    this.apiServer = null;
    this.rl = null;
    this.apiHost = '127.0.0.1';
    this.apiPort = 8000;
    this.apiBaseUrl = `http://${this.apiHost}:${this.apiPort}`;
    this.isLogging = false;  // Flag to prevent recursive logging
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

  async logServerOutput(data) {
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
          const logEntry = `[${timestamp}] [PID:${pid || 'unknown'}] ${trimmedLine}\n`;
          fs.appendFileSync('server.log', logEntry);
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
        stdio: ['ignore', 'pipe', 'pipe']  // Pipe stdout and stderr for logging
      });

      // Log server output to file with timestamp and PID
      this.apiServer.stdout.on('data', (data) => {
        this.logServerOutput(data.toString());
      });

      this.apiServer.stderr.on('data', (data) => {
        this.logServerOutput(data.toString());
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
      } else {
        console.log(chalk.red('Error: ' + (response.error || 'Command failed')));
      }
    } catch (err) {
      console.log(chalk.red('✗ Request failed: ' + err.message));
    }

    this.rl.prompt();
  }

  handleOutput(output) {
    const lines = output.split('\n');

    lines.forEach(line => {
      if (line && !line.includes('pmshell>')) {
        // Format different types of output
        if (line.startsWith('***')) {
          // Error messages
          console.log(chalk.red(line));
        } else if (line.startsWith('Model:')) {
          // Model info
          console.log(chalk.cyan.bold(line));
        } else if (line.includes('=')) {
          // Config/parameter lines
          const parts = line.split('=');
          if (parts.length >= 2) {
            const key = parts[0].trim();
            const value = parts.slice(1).join('=').trim();
            console.log(chalk.gray(key) + chalk.white(' = ') + chalk.yellow(value));
          } else {
            console.log(line);
          }
        } else if (line.startsWith('LLM provider:')) {
          // LLM section header
          console.log(chalk.magenta.bold(line));
        } else if (line.trim().startsWith('Name:') || line.trim().startsWith('Model:')) {
          // LLM details
          console.log(chalk.gray(line));
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
      const url = this.apiBaseUrl + endpoint;
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

  async showHelp() {
    console.log(chalk.cyan.bold('\nPMShell Client Commands:'));
    console.log(chalk.gray('─'.repeat(50)));

    console.log(chalk.green('  help, ?') + '               ' + chalk.white('Show this help message'));
    console.log(chalk.green('  kill server') + '           ' + chalk.white('Terminate the backend API server'));
    console.log(chalk.green('  restart server') + '        ' + chalk.white('Restart the backend API server'));
    console.log(chalk.green('  quit, exit') + '            ' + chalk.white('Exit the client'));

    // Get pmshell help from backend
    try {
      const response = await this.makeRequest('/command', {
        method: 'POST',
        body: JSON.stringify({ command: 'help' })
      });

      if (response.success && response.output) {
        console.log(chalk.cyan.bold('\nPMShell Server Commands:'));
        console.log(chalk.gray('─'.repeat(50)));
        console.log(response.output);
      }
    } catch (err) {
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
    console.log(chalk.green('✓ Server restarted successfully\n'));
  }

  async shutdown() {
    console.log(chalk.cyan('\nShutting down client...'));

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

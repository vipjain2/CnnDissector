#!/usr/bin/env node

const { spawn } = require('child_process');
const { createInterface } = require('readline');
const chalk = require('chalk');
const boxen = require('boxen');
const http = require('http');

class PMShellAPIFrontend {
  constructor() {
    this.apiServer = null;
    this.serverPid = null;
    this.rl = null;
    this.apiHost = '127.0.0.1';
    this.apiPort = 8000;
    this.apiBaseUrl = `http://${this.apiHost}:${this.apiPort}`;
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

    console.log(chalk.gray('Starting API backend server...'));

    // Start the API server
    await this.startAPIServer();

    // Wait for server to be ready
    await this.waitForServer();

    console.log(chalk.green('✓ Backend server ready\n'));

    // Setup readline interface
    this.setupReadline();
  }

  startAPIServer() {
    const serverPath = '../src/pm_api_server.py';

    return new Promise((resolve, reject) => {
      // Run the API server as detached process
      this.apiServer = spawn(serverPath, [
        '--host', this.apiHost,
        '--port', this.apiPort.toString()
      ], {
        env: process.env,
        detached: true,  // Run independently
        stdio: 'ignore'  // Don't pipe stdio
      });

      // Store the PID for later use
      this.serverPid = this.apiServer.pid;

      // Detach the process so it can run independently
      this.apiServer.unref();

      // Handle process exit - don't crash client
      this.apiServer.on('close', (code) => {
        if (code !== 0) {
          console.log(chalk.yellow(`\n⚠ Backend server exited with code ${code}`));
          console.log(chalk.gray('The client will continue running, but API calls will fail.'));
          console.log(chalk.gray('You may need to restart the backend server manually.'));
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

      if (input === 'quit' || input === 'exit') {
        await this.shutdown();
        return;
      }

      if (input === 'kill_server') {
        await this.killServer();
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

  async killServer() {
    if (!this.serverPid) {
      console.log(chalk.yellow('⚠ No server PID stored. Server may not be running.'));
      return;
    }

    try {
      // Kill the server process
      process.kill(this.serverPid, 'SIGTERM');
      console.log(chalk.green(`✓ Sent SIGTERM to backend server (PID: ${this.serverPid})`));
      this.serverPid = null;
    } catch (err) {
      if (err.code === 'ESRCH') {
        console.log(chalk.yellow(`⚠ Server process (PID: ${this.serverPid}) not found. It may have already exited.`));
        this.serverPid = null;
      } else {
        console.log(chalk.red(`✗ Failed to kill server: ${err.message}`));
      }
    }
  }

  async shutdown() {
    console.log(chalk.cyan('\nShutting down client...'));

    // Note: API server is detached and will continue running
    // To stop it, you need to manually kill the process or use an API endpoint

    if (this.rl) {
      this.rl.close();
    }

    console.log(chalk.gray('Note: Backend server is still running. Use "kill_server" command to stop it.'));
    process.exit(0);
  }
}

// Start the frontend
const frontend = new PMShellAPIFrontend();
frontend.start().catch(err => {
  console.error(chalk.red('Failed to start:', err.message));
  process.exit(1);
});

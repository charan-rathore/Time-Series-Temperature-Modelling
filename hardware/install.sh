#!/bin/bash
# ThermoSense Sensor Installation Script for Raspberry Pi
#
# This script:
# 1. Installs Python dependencies
# 2. Sets up systemd services for sensor daemon and uploader
# 3. Enables services to start on boot
#
# Usage:
#   chmod +x install.sh
#   sudo ./install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_USER="${SUDO_USER:-pi}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=============================================="
echo "  ThermoSense Sensor Installation"
echo "=============================================="
echo ""
echo "Script directory: $SCRIPT_DIR"
echo "Service user: $SERVICE_USER"
echo "Python: $PYTHON_BIN"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "[ERROR] Please run with sudo: sudo ./install.sh"
    exit 1
fi

# Check Python version
echo "[1/6] Checking Python version..."
PYTHON_VERSION=$($PYTHON_BIN --version 2>&1 | grep -oP '\d+\.\d+')
echo "       Found Python $PYTHON_VERSION"

# Install system dependencies for Adafruit_DHT
echo "[2/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3-pip python3-dev libgpiod2

# Install Python dependencies
echo "[3/6] Installing Python dependencies..."
$PYTHON_BIN -m pip install --quiet --upgrade pip
$PYTHON_BIN -m pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# Create sensor daemon service file
echo "[4/6] Creating sensor daemon service..."
cat > /etc/systemd/system/thermosense-sensor.service << EOF
[Unit]
Description=ThermoSense Sensor Daemon
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$PYTHON_BIN $SCRIPT_DIR/sensor_daemon.py --interval 300 --http-port 8081
Restart=always
RestartSec=10
StandardOutput=append:$SCRIPT_DIR/sensor_daemon.log
StandardError=append:$SCRIPT_DIR/sensor_daemon.log

# Environment variables
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOF

# Create uploader service file
echo "[5/6] Creating uploader service..."
cat > /etc/systemd/system/thermosense-uploader.service << EOF
[Unit]
Description=ThermoSense Data Uploader
After=network-online.target thermosense-sensor.service
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$PYTHON_BIN $SCRIPT_DIR/uploader.py --continuous --interval 900
Restart=always
RestartSec=60
StandardOutput=append:$SCRIPT_DIR/uploader.log
StandardError=append:$SCRIPT_DIR/uploader.log

# Environment variables (edit these!)
Environment="PYTHONUNBUFFERED=1"
Environment="THERMOSENSE_API_URL=http://localhost:8000"
# Environment="THERMOSENSE_API_KEY=your_api_key_here"

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
echo "[6/6] Enabling services..."
systemctl daemon-reload
systemctl enable thermosense-sensor.service
systemctl enable thermosense-uploader.service

echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "Services installed but NOT started yet."
echo ""
echo "To start the sensor daemon:"
echo "  sudo systemctl start thermosense-sensor"
echo ""
echo "To start the uploader (after configuring API URL):"
echo "  sudo systemctl edit thermosense-uploader  # Set THERMOSENSE_API_URL"
echo "  sudo systemctl start thermosense-uploader"
echo ""
echo "To check status:"
echo "  systemctl status thermosense-sensor"
echo "  systemctl status thermosense-uploader"
echo ""
echo "To view logs:"
echo "  journalctl -u thermosense-sensor -f"
echo "  tail -f $SCRIPT_DIR/sensor_daemon.log"
echo ""
echo "Test the sensor daemon:"
echo "  curl http://localhost:8081/health"
echo "  curl http://localhost:8081/latest"
echo ""

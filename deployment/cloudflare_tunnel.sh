#!/bin/bash
# ThermoSense Cloudflare Tunnel Setup for Edge Deployment
#
# This script sets up Cloudflare Tunnel to expose the ThermoSense API
# running on a Raspberry Pi or local machine to the internet.
#
# Prerequisites:
#   1. Cloudflare account (free)
#   2. cloudflared CLI installed: 
#      - Raspberry Pi: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o cloudflared
#      - macOS: brew install cloudflared
#      - Linux x64: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
#
# Usage:
#   # Quick tunnel (temporary URL, good for testing)
#   ./cloudflare_tunnel.sh quick
#
#   # Named tunnel (persistent URL, requires Cloudflare login)
#   ./cloudflare_tunnel.sh setup    # First time: login and create tunnel
#   ./cloudflare_tunnel.sh start    # Run the tunnel
#   ./cloudflare_tunnel.sh install  # Install as systemd service

set -e

TUNNEL_NAME="thermosense"
LOCAL_PORT="${THERMOSENSE_PORT:-8000}"
LOCAL_URL="http://localhost:$LOCAL_PORT"

check_cloudflared() {
    if ! command -v cloudflared &> /dev/null; then
        echo "Error: cloudflared not installed."
        echo ""
        echo "Install instructions:"
        echo "  Raspberry Pi (ARM64):"
        echo "    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o cloudflared"
        echo "    chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/"
        echo ""
        echo "  macOS:"
        echo "    brew install cloudflared"
        echo ""
        echo "  Linux (x64):"
        echo "    curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared"
        echo "    chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/"
        exit 1
    fi
}

check_api() {
    if ! curl -sf "$LOCAL_URL/api/health" > /dev/null 2>&1; then
        echo "Warning: ThermoSense API not responding at $LOCAL_URL"
        echo "Make sure the API is running: uvicorn src.api.main:app --port $LOCAL_PORT"
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

quick_tunnel() {
    echo "Starting quick tunnel (temporary URL)..."
    echo "This URL will change each time. For a persistent URL, use: $0 setup"
    echo ""
    check_api
    cloudflared tunnel --url "$LOCAL_URL"
}

setup_tunnel() {
    echo "Setting up named tunnel: $TUNNEL_NAME"
    echo ""
    
    # Check if already logged in
    if ! cloudflared tunnel list &> /dev/null; then
        echo "Logging in to Cloudflare..."
        cloudflared tunnel login
    fi
    
    # Check if tunnel exists
    if cloudflared tunnel list | grep -q "$TUNNEL_NAME"; then
        echo "Tunnel '$TUNNEL_NAME' already exists."
    else
        echo "Creating tunnel '$TUNNEL_NAME'..."
        cloudflared tunnel create "$TUNNEL_NAME"
    fi
    
    # Get tunnel UUID
    TUNNEL_UUID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
    
    echo ""
    echo "Tunnel created!"
    echo "  Name: $TUNNEL_NAME"
    echo "  UUID: $TUNNEL_UUID"
    echo ""
    echo "Next steps:"
    echo "  1. Run the tunnel: $0 start"
    echo "  2. Or install as service: $0 install"
    echo ""
    echo "Your tunnel URL will be: https://$TUNNEL_NAME.cfargotunnel.com"
    echo "(Or configure a custom domain in Cloudflare dashboard)"
}

start_tunnel() {
    echo "Starting tunnel: $TUNNEL_NAME -> $LOCAL_URL"
    check_api
    cloudflared tunnel --url "$LOCAL_URL" run "$TUNNEL_NAME"
}

install_service() {
    echo "Installing tunnel as systemd service..."
    
    # Create config file
    CONFIG_DIR="$HOME/.cloudflared"
    mkdir -p "$CONFIG_DIR"
    
    cat > "$CONFIG_DIR/config.yml" << EOF
tunnel: $TUNNEL_NAME
credentials-file: $CONFIG_DIR/credentials.json

ingress:
  - service: $LOCAL_URL
EOF
    
    # Install service
    sudo cloudflared service install
    
    echo ""
    echo "Service installed! Commands:"
    echo "  sudo systemctl start cloudflared"
    echo "  sudo systemctl enable cloudflared"
    echo "  sudo systemctl status cloudflared"
}

show_help() {
    echo "ThermoSense Cloudflare Tunnel Setup"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  quick    Start a quick tunnel with temporary URL (no login required)"
    echo "  setup    Login to Cloudflare and create a named tunnel"
    echo "  start    Run an existing named tunnel"
    echo "  install  Install tunnel as systemd service (for Raspberry Pi)"
    echo "  help     Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  THERMOSENSE_PORT  Local API port (default: 8000)"
}

# Main
check_cloudflared

case "${1:-help}" in
    quick)
        quick_tunnel
        ;;
    setup)
        setup_tunnel
        ;;
    start)
        start_tunnel
        ;;
    install)
        install_service
        ;;
    help|*)
        show_help
        ;;
esac

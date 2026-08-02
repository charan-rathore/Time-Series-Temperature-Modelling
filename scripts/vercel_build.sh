#!/usr/bin/env bash
# Build the React dashboard and stage it for Vercel CDN (public/) + FastAPI fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/frontend"

# Prefer a clean lockfile install; fall back if the lock is out of sync.
if [ -f package-lock.json ]; then
  npm ci || npm install
else
  npm install
fi

npm run build

# Serve static assets from Vercel CDN
rm -rf "$ROOT/public"
mkdir -p "$ROOT/public"
cp -R build/. "$ROOT/public/"

echo "[vercel_build] Frontend built → frontend/build and public/"

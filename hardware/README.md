# ThermoSense Hardware — Raspberry Pi Sensor Setup

This directory contains code for deploying a real temperature/humidity sensor using a Raspberry Pi and DHT22.

## Hardware Required

| Component | Approx. Cost | Notes |
|-----------|-------------|-------|
| Raspberry Pi Zero 2 W | $15 | Any Pi with GPIO works |
| DHT22 sensor module | $5 | AM2302 variant recommended (has pull-up resistor) |
| Jumper wires | $3 | Female-to-female for Pi GPIO |
| (Optional) Breadboard | $5 | For prototyping |

**Total: ~$25**

## Wiring Diagram

```
DHT22 Module        Raspberry Pi
─────────────       ─────────────
   VCC (1) ───────── 3.3V (Pin 1)
   DATA (2) ──────── GPIO4 (Pin 7)
   NC (3)            (not connected)
   GND (4) ───────── GND (Pin 9)
```

**Note**: If using a bare DHT22 sensor (not a module), add a 10kΩ pull-up resistor between VCC and DATA.

## Quick Start (Raspberry Pi)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/Time-Series-Temperature-Modelling.git
cd Time-Series-Temperature-Modelling/hardware

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Test the sensor (reads once and exits)
python3 -c "import Adafruit_DHT; print(Adafruit_DHT.read_retry(Adafruit_DHT.DHT22, 4))"

# 4. Start the daemon manually (for testing)
python3 sensor_daemon.py --interval 60

# 5. In another terminal, test the HTTP endpoint
curl http://localhost:8081/health
curl http://localhost:8081/latest

# 6. Install as system service (production)
sudo ./install.sh
```

## Files

| File | Description |
|------|-------------|
| `sensor_daemon.py` | Main daemon: reads sensor, stores to SQLite, exposes HTTP API |
| `uploader.py` | Syncs local readings to the ThermoSense cloud API |
| `install.sh` | Installs systemd services for auto-start |
| `requirements.txt` | Python dependencies |

## Sensor Daemon HTTP API

The daemon exposes a simple HTTP API on port 8081 (configurable):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check daemon status |
| `/latest` | GET | Get most recent reading |
| `/unsynced` | GET | Get readings not yet synced to cloud |
| `/stats` | GET | Get 24-hour statistics |
| `/9pm?date=YYYY-MM-DD` | GET | Get the 9 PM reading for a specific date |
| `/mark_synced` | POST | Mark readings as synced (used by uploader) |

### Example Responses

```bash
# Health check
$ curl http://localhost:8081/health
{
  "status": "ok",
  "sensor_available": true
}

# Latest reading
$ curl http://localhost:8081/latest
{
  "id": 1234,
  "timestamp": "2026-05-07T15:30:00Z",
  "temp_c": 28.5,
  "humidity_pct": 65.2,
  "source": "dht22_sensor",
  "synced": 0
}

# 24-hour stats
$ curl http://localhost:8081/stats
{
  "period_hours": 24,
  "reading_count": 288,
  "avg_temp_c": 27.3,
  "min_temp_c": 24.1,
  "max_temp_c": 31.2,
  "avg_humidity_pct": 68.5,
  "unsynced_count": 12
}
```

## Simulation Mode (No Hardware)

For development/testing without a Raspberry Pi, the daemon runs in simulation mode:

```bash
# Force simulation mode
python3 sensor_daemon.py --simulate

# Simulated readings follow a realistic diurnal pattern
```

## Deployment Checklist

- [ ] Pi connected to WiFi with stable internet
- [ ] DHT22 wired correctly (see diagram above)
- [ ] `sensor_daemon.py` running and returning readings
- [ ] SQLite database being populated (`readings.db`)
- [ ] Uploader configured with correct cloud API URL
- [ ] Both services enabled for auto-start on boot
- [ ] Sensor placed outdoors, sheltered from rain/sun
- [ ] GPS coordinates noted for Open-Meteo comparison

## Troubleshooting

### "Failed to read sensor"

1. Check wiring — data pin should be on GPIO4
2. Verify sensor power (3.3V or 5V)
3. Try adding a 10kΩ pull-up resistor
4. Run `gpio readall` to check GPIO status
5. Try different GPIO pin and update `--gpio` flag

### "Adafruit_DHT not available"

```bash
# Install on Raspberry Pi OS
sudo apt-get install python3-dev python3-pip
pip3 install Adafruit-DHT
```

### Permission denied on GPIO

```bash
# Add user to gpio group
sudo usermod -aG gpio $USER
# Log out and back in
```

### Uploader can't reach cloud API

1. Check `THERMOSENSE_API_URL` environment variable
2. Verify network connectivity: `curl http://your-api-url/api/health`
3. Check firewall rules if applicable

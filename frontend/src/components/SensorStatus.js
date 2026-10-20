import React, { useEffect, useState } from "react";
import { api } from "../api";

/**
 * Live sensor freshness, in plain lowercase words.
 * Renders nothing until the API answers; never invents a reading.
 */
export default function SensorStatus() {
  const [reading, setReading] = useState(null);
  const [settled, setSettled] = useState(false);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    if (typeof api.getSensorLatest !== "function") return undefined;
    let cancelled = false;
    const load = () => {
      api
        .getSensorLatest()
        .then((r) => {
          if (!cancelled) {
            setReading(r);
            setUnreachable(false);
            setSettled(true);
          }
        })
        .catch((e) => {
          if (cancelled) return;
          setReading(null);
          setUnreachable(!/no sensor data|404/i.test(e?.message || ""));
          setSettled(true);
        });
    };
    load();
    const timer = setInterval(load, 5 * 60 * 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  if (!settled) return null;

  let text;
  let state = "idle";
  if (unreachable) {
    text = "sensor · status unavailable right now";
  } else if (!reading) {
    text = "sensor · waiting for its first reading";
  } else {
    const temp =
      typeof reading.sensor_temp_c === "number"
        ? `${reading.sensor_temp_c.toFixed(1)}°c`
        : null;
    const day = reading.date; // YYYY-MM-DD of the 9 PM reading
    const fmt = (d) =>
      `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    const now = new Date();
    if (day === fmt(now)) {
      state = "live";
      text = `sensor · ${temp ? `${temp} ` : ""}this evening`;
    } else if (day === fmt(new Date(now.getTime() - 86400000))) {
      state = "live";
      text = `sensor · ${temp ? `${temp} ` : ""}yesterday evening`;
    } else {
      state = "stale";
      text = `sensor · quiet since ${day || "a while ago"}`;
    }
  }

  return (
    <p className={`sensor-status sensor-${state}`}>
      <span className="sensor-dot" aria-hidden="true" />
      {text}
    </p>
  );
}

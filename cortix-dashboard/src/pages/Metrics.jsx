import React, { useState, useEffect } from "react";
import axios from "axios";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

function Metrics() {
  const [metrics, setMetrics] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:8000/api/metrics")
      .then((res) => setMetrics(res.data))
      .catch(() => {});
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      <div>
        <h2 style={{ fontSize: "28px", fontWeight: "bold", margin: "0 0 8px 0" }}>System Performance Charts</h2>
        <p style={{ color: "#9ca3af", margin: 0 }}>Real-time False Positive Rate (FPR), processing latency, and packet throughput timelines.</p>
      </div>

      <div style={{
        backgroundColor: "#11141e",
        borderRadius: "16px",
        border: "1px solid #1f2937",
        padding: "24px"
      }}>
        <h3 style={{ fontSize: "20px", fontWeight: "bold", margin: "0 0 20px 0" }}>Processing Latency over Time (ms)</h3>
        {metrics.length === 0 ? (
          <div style={{ color: "#6b7280", textAlign: "center", padding: "40px" }}>
            Waiting for metrics data... Start the daemon to begin collecting performance telemetry.
          </div>
        ) : (
          <div style={{ width: "100%", height: 350 }}>
            <ResponsiveContainer>
              <LineChart data={metrics}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="timestamp" stroke="#9ca3af" tickFormatter={(tick) => new Date(tick).toLocaleTimeString()} />
                <YAxis stroke="#9ca3af" />
                <Tooltip contentStyle={{ backgroundColor: "#11141e", borderColor: "#374151" }} />
                <Line type="monotone" dataKey="latency_p50_ms" stroke="#3b82f6" strokeWidth={3} name="p50 Latency" dot={false} />
                <Line type="monotone" dataKey="latency_p99_ms" stroke="#ef4444" strokeWidth={2} name="p99 Latency" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
}

export default Metrics;

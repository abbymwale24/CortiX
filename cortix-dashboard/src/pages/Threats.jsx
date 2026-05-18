import React, { useState, useEffect } from "react";
import axios from "axios";

function Threats() {
  const [threats, setThreats] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:8000/api/threats?limit=50")
      .then((res) => setThreats(res.data))
      .catch((err) => console.debug("Offline API mock"));
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      <div>
        <h2 style={{ fontSize: "28px", fontWeight: "bold", margin: "0 0 8px 0" }}>Threat Activity Feed</h2>
        <p style={{ color: "#9ca3af", margin: 0 }}>Detailed log of all anomalous events flagged by the Spiking Neural Network and LSTM-CNN model.</p>
      </div>

      <div style={{
        backgroundColor: "#11141e",
        borderRadius: "16px",
        border: "1px solid #1f2937",
        padding: "24px"
      }}>
        <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #1f2937", color: "#6b7280" }}>
              <th style={{ padding: "12px 16px" }}>TIMESTAMP</th>
              <th style={{ padding: "12px 16px" }}>SOURCE IP</th>
              <th style={{ padding: "12px 16px" }}>DESTINATION</th>
              <th style={{ padding: "12px 16px" }}>PROTOCOL</th>
              <th style={{ padding: "12px 16px" }}>ATTACK CLASS</th>
              <th style={{ padding: "12px 16px" }}>CONFIDENCE</th>
              <th style={{ padding: "12px 16px" }}>SNN Z-SCORE</th>
              <th style={{ padding: "12px 16px" }}>ACTION TAKEN</th>
            </tr>
          </thead>
          <tbody>
            {(threats.length > 0 ? threats : sampleThreats).map((threat, idx) => (
              <tr key={idx} style={{ borderBottom: "1px solid #1f2937", color: "#d1d5db" }}>
                <td style={{ padding: "16px" }}>{new Date(threat.timestamp || Date.now()).toLocaleString()}</td>
                <td style={{ padding: "16px", fontWeight: "bold" }}>{threat.src_ip}</td>
                <td style={{ padding: "16px" }}>{threat.dst_ip}:{threat.dst_port}</td>
                <td style={{ padding: "16px" }}>{threat.protocol}</td>
                <td style={{ padding: "16px" }}>
                  <span style={badgeStyle(threat.attack_class)}>{threat.attack_class}</span>
                </td>
                <td style={{ padding: "16px" }}>{(threat.confidence * 100).toFixed(1)}%</td>
                <td style={{ padding: "16px" }}>{threat.z_score.toFixed(2)}</td>
                <td style={{ padding: "16px", color: "#ef4444", fontWeight: "bold" }}>{threat.action_taken}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const badgeStyle = (cls) => {
  const colors = {
    BENIGN: { bg: "#ecfdf5", text: "#065f46" },
    DoS: { bg: "#fef2f2", text: "#991b1b" },
    DDoS: { bg: "#fff5f5", text: "#c53030" },
    PortScan: { bg: "#fffbeb", text: "#92400e" },
    BruteForce: { bg: "#f3e8ff", text: "#6b21a8" },
  };
  const val = colors[cls] || { bg: "#eff6ff", text: "#1e40af" };
  return {
    padding: "6px 12px",
    borderRadius: "20px",
    fontSize: "12px",
    fontWeight: "600",
    backgroundColor: val.bg,
    color: val.text
  };
};

const sampleThreats = [
  { timestamp: new Date().toISOString(), src_ip: "10.0.0.1", dst_ip: "10.0.0.2", dst_port: 80, protocol: "TCP", attack_class: "PortScan", confidence: 0.985, z_score: 5.4, action_taken: "TEMP_BLOCK" },
  { timestamp: new Date().toISOString(), src_ip: "185.12.5.4", dst_ip: "10.0.0.3", dst_port: 443, protocol: "TCP", attack_class: "DoS", confidence: 0.992, z_score: 11.2, action_taken: "HARD_BLOCK" },
  { timestamp: new Date().toISOString(), src_ip: "192.168.1.12", dst_ip: "10.0.0.2", dst_port: 22, protocol: "TCP", attack_class: "BENIGN", confidence: 1.0, z_score: 1.1, action_taken: "ALLOW" },
];

export default Threats;

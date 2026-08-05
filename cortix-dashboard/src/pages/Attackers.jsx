import React, { useState, useEffect } from "react";
import axios from "axios";
import { Globe, MapPin, Compass } from "lucide-react";

function Attackers() {
  const [attackers, setAttackers] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:8000/api/attackers")
      .then((res) => setAttackers(res.data))
      .catch(() => {});
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      <div>
        <h2 style={{ fontSize: "28px", fontWeight: "bold", margin: "0 0 8px 0" }}>Attacker Attribution Profiles</h2>
        <p style={{ color: "#9ca3af", margin: 0 }}>Passive OSINT-derived profiles of identified adversaries.</p>
      </div>

      {attackers.length === 0 ? (
        <div style={{
          backgroundColor: "#11141e",
          borderRadius: "16px",
          border: "1px solid #1f2937",
          padding: "40px",
          textAlign: "center",
          color: "#6b7280",
        }}>
          No attacker profiles recorded yet. Profiles are created when threats are detected.
        </div>
      ) : (
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "24px"
        }}>
          {attackers.map((attacker, idx) => (
            <div key={idx} style={{
              backgroundColor: "#11141e",
              borderRadius: "16px",
              border: "1px solid #1f2937",
              padding: "24px",
              display: "flex",
              flexDirection: "column",
              gap: "16px",
              boxShadow: "0 4px 20px rgba(0, 0, 0, 0.4)"
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontSize: "18px", fontWeight: "bold", color: "#ef4444" }}>{attacker.ip}</div>
                <span style={threatBadgeStyle(attacker.threat_level)}>{attacker.threat_level}</span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "8px", borderTop: "1px solid #1f2937", paddingTop: "16px" }}>
                <div style={infoRow}><MapPin size={16} color="#9ca3af" /> <strong>Location:</strong> {attacker.city}, {attacker.country}</div>
                <div style={infoRow}><Globe size={16} color="#9ca3af" /> <strong>ISP/ASN:</strong> {attacker.isp} ({attacker.asn})</div>
                <div style={infoRow}><Compass size={16} color="#9ca3af" /> <strong>Reverse DNS:</strong> {attacker.hostname}</div>
              </div>

              <div style={{
                display: "flex",
                justifyContent: "space-between",
                backgroundColor: "#161b26",
                padding: "12px",
                borderRadius: "8px",
                fontSize: "13px"
              }}>
                <div>Abuse Score: <strong style={{ color: "#eab308" }}>{attacker.abuse_score}/100</strong></div>
                <div>VT Verdicts: <strong style={{ color: "#ef4444" }}>{attacker.vt_malicious}</strong></div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const infoRow = {
  display: "flex",
  alignItems: "center",
  gap: "10px",
  fontSize: "14px",
  color: "#d1d5db"
};

const threatBadgeStyle = (lvl) => {
  const colors = {
    CRITICAL: { bg: "#fef2f2", text: "#991b1b" },
    HIGH: { bg: "#fffbeb", text: "#92400e" },
    MEDIUM: { bg: "#eff6ff", text: "#1e40af" },
  };
  const val = colors[lvl] || { bg: "#f3f4f6", text: "#374151" };
  return {
    padding: "4px 8px",
    borderRadius: "4px",
    fontSize: "11px",
    fontWeight: "bold",
    backgroundColor: val.bg,
    color: val.text
  };
};

export default Attackers;

import React, { useState, useEffect } from "react";
import axios from "axios";
import { ShieldCheck, ShieldAlert, RotateCcw } from "lucide-react";

function Containment() {
  const [actions, setActions] = useState([]);
  const [form, setForm] = useState({ src_ip: "", action_id: 1 });

  useEffect(() => {
    fetchActions();
  }, []);

  const fetchActions = () => {
    axios.get("http://localhost:8000/api/containment")
      .then((res) => setActions(res.data))
      .catch((err) => console.debug("Offline API mock"));
  };

  const handleApply = (e) => {
    e.preventDefault();
    axios.post("http://localhost:8000/api/containment", form)
      .then(() => {
        fetchActions();
        setForm({ src_ip: "", action_id: 1 });
      })
      .catch((err) => alert("Firewall override request failed. Ensure backend API is active."));
  };

  const handleRestore = (ip) => {
    axios.post(`http://localhost:8000/api/containment/restore?src_ip=${ip}`)
      .then(() => fetchActions())
      .catch((err) => alert("Failed to lift rules."));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
      <div>
        <h2 style={{ fontSize: "28px", fontWeight: "bold", margin: "0 0 8px 0" }}>Active Containment Center</h2>
        <p style={{ color: "#9ca3af", margin: 0 }}>Configure and audit live blockages, rate-limits, and honeypot redirects.</p>
      </div>

      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 2fr",
        gap: "32px",
        alignItems: "start"
      }}>
        {/* Manual Rules Override Form */}
        <form onSubmit={handleApply} style={{
          backgroundColor: "#11141e",
          borderRadius: "16px",
          border: "1px solid #1f2937",
          padding: "24px",
          display: "flex",
          flexDirection: "column",
          gap: "16px"
        }}>
          <h3 style={{ fontSize: "18px", fontWeight: "bold", margin: "0 0 10px 0" }}>Manual Rule Override</h3>

          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label style={{ fontSize: "13px", color: "#9ca3af" }}>Target IP Address</label>
            <input
              type="text"
              required
              value={form.src_ip}
              onChange={(e) => setForm({ ...form, src_ip: e.target.value })}
              placeholder="e.g. 10.0.0.1"
              style={inputStyle}
            />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <label style={{ fontSize: "13px", color: "#9ca3af" }}>Select Rule Action</label>
            <select
              value={form.action_id}
              onChange={(e) => setForm({ ...form, action_id: parseInt(e.target.value) })}
              style={inputStyle}
            >
              <option value={1}>RATE_LIMIT (Throttle to 10%)</option>
              <option value={2}>TEMP_BLOCK (60 seconds drop)</option>
              <option value={3}>QUARANTINE (Redirect to VLAN 999)</option>
              <option value={4}>HARD_BLOCK (Permanent ACL drop)</option>
              <option value={5}>HONEYPOT_REDIRECT (Docker forward)</option>
            </select>
          </div>

          <button type="submit" style={{
            backgroundColor: "#ef4444",
            color: "#ffffff",
            border: "none",
            borderRadius: "8px",
            padding: "12px",
            fontWeight: "bold",
            cursor: "pointer",
            marginTop: "10px",
            transition: "background-color 0.2s"
          }}>
            Apply Firewall Action
          </button>
        </form>

        {/* Action Logs List */}
        <div style={{
          backgroundColor: "#11141e",
          borderRadius: "16px",
          border: "1px solid #1f2937",
          padding: "24px"
        }}>
          <h3 style={{ fontSize: "18px", fontWeight: "bold", margin: "0 0 20px 0" }}>Active Rule Auditing</h3>
          <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #1f2937", color: "#6b7280" }}>
                <th style={{ padding: "12px 16px" }}>IP ADDRESS</th>
                <th style={{ padding: "12px 16px" }}>RULE TYPE</th>
                <th style={{ padding: "12px 16px" }}>SOURCE</th>
                <th style={{ padding: "12px 16px" }}>STATUS</th>
                <th style={{ padding: "12px 16px" }}>RESTORE</th>
              </tr>
            </thead>
            <tbody>
              {(actions.length > 0 ? actions : sampleActions).map((act, idx) => (
                <tr key={idx} style={{ borderBottom: "1px solid #1f2937", color: "#d1d5db" }}>
                  <td style={{ padding: "16px", fontWeight: "bold" }}>{act.src_ip}</td>
                  <td style={{ padding: "16px" }}>{act.action}</td>
                  <td style={{ padding: "16px" }}>{act.triggered_by}</td>
                  <td style={{ padding: "16px" }}>
                    {act.resolved ? (
                      <span style={{ color: "#10b981", display: "flex", alignItems: "center", gap: "6px", fontSize: "13px" }}><ShieldCheck size={16} /> Restored</span>
                    ) : (
                      <span style={{ color: "#ef4444", display: "flex", alignItems: "center", gap: "6px", fontSize: "13px" }}><ShieldAlert size={16} /> Active</span>
                    )}
                  </td>
                  <td style={{ padding: "16px" }}>
                    {!act.resolved && (
                      <button onClick={() => handleRestore(act.src_ip)} style={{
                        background: "none",
                        border: "none",
                        color: "#3b82f6",
                        cursor: "pointer",
                        display: "flex",
                        alignItems: "center",
                        gap: "6px"
                      }}>
                        <RotateCcw size={16} /> Restore
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

const inputStyle = {
  backgroundColor: "#161b26",
  border: "1px solid #374151",
  borderRadius: "8px",
  color: "#ffffff",
  padding: "10px 14px",
  fontSize: "14px",
  outline: "none"
};

const sampleActions = [
  { src_ip: "10.0.0.1", action: "TEMP_BLOCK", triggered_by: "RL_AGENT", resolved: false },
  { src_ip: "185.12.5.4", action: "HARD_BLOCK", triggered_by: "ADMIN", resolved: false },
  { src_ip: "192.168.1.12", action: "RATE_LIMIT", triggered_by: "RL_AGENT", resolved: true },
];

export default Containment;

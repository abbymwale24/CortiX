import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import { Shield, AlertTriangle, Users, BarChart3, Settings, ShieldAlert, Activity } from "lucide-react";
import Overview from "./pages/Overview";
import Threats from "./pages/Threats";
import Attackers from "./pages/Attackers";
import Metrics from "./pages/Metrics";
import Containment from "./pages/Containment";

function App() {
  const [liveAlerts, setLiveAlerts] = useState([]);

  useEffect(() => {
    // Connect to FastAPI live WebSocket stream
    const ws = new WebSocket("ws://localhost:8000/ws/live");
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.event === "THREAT_ALERT") {
          setLiveAlerts((prev) => [data, ...prev].slice(0, 10));
        }
      } catch (err) {
        console.error("Failed to parse websocket update:", err);
      }
    };
    return () => ws.close();
  }, []);

  return (
    <Router>
      <div style={{
        display: "flex",
        minHeight: "100vh",
        backgroundColor: "#0d0f14",
        color: "#f3f4f6",
        fontFamily: "'Outfit', sans-serif"
      }}>
        {/* Navigation Sidebar */}
        <aside style={{
          width: "280px",
          backgroundColor: "#11141e",
          borderRight: "1px solid #1f2937",
          padding: "24px",
          display: "flex",
          flexDirection: "column"
        }}>
          {/* Logo Branding */}
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "40px" }}>
            <div style={{
              background: "linear-gradient(135deg, #ef4444, #b91c1c)",
              padding: "10px",
              borderRadius: "12px",
              boxShadow: "0 0 15px rgba(239, 68, 68, 0.4)"
            }}>
              <ShieldAlert size={28} color="#ffffff" />
            </div>
            <div>
              <h1 style={{ fontSize: "22px", fontWeight: "bold", margin: 0, tracking: "0.05em", color: "#ffffff" }}>CORTIX</h1>
              <span style={{ fontSize: "11px", color: "#6b7280", letterSpacing: "1px" }}>NEURO DEFENDER</span>
            </div>
          </div>

          {/* Nav List */}
          <nav style={{ display: "flex", flexDirection: "column", gap: "8px", flex: 1 }}>
            <Link to="/" style={navItemStyle}><Shield size={20} /> Overview</Link>
            <Link to="/threats" style={navItemStyle}><AlertTriangle size={20} /> Threat Feed</Link>
            <Link to="/attackers" style={navItemStyle}><Users size={20} /> Attacker Maps</Link>
            <Link to="/metrics" style={navItemStyle}><BarChart3 size={20} /> Brain Metrics</Link>
            <Link to="/containment" style={navItemStyle}><Activity size={20} /> Firewall Rules</Link>
          </nav>

          {/* Footer stats badge */}
          <div style={{
            marginTop: "auto",
            backgroundColor: "#161b26",
            padding: "16px",
            borderRadius: "12px",
            border: "1px solid #374151",
            display: "flex",
            alignItems: "center",
            gap: "12px"
          }}>
            <div style={{
              width: "12px",
              height: "12px",
              borderRadius: "50%",
              backgroundColor: "#10b981",
              boxShadow: "0 0 8px #10b981"
            }} />
            <div>
              <div style={{ fontSize: "12px", fontWeight: "bold" }}>SYSTEM ACTIVE</div>
              <div style={{ fontSize: "10px", color: "#6b7280" }}>FPR: 0.02% | p50: 4.2ms</div>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main style={{ flex: 1, padding: "40px", overflowY: "auto" }}>
          <Routes>
            <Route path="/" element={<Overview liveAlerts={liveAlerts} />} />
            <Route path="/threats" element={<Threats />} />
            <Route path="/attackers" element={<Attackers />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/containment" element={<Containment />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

const navItemStyle = {
  display: "flex",
  alignItems: "center",
  gap: "12px",
  padding: "12px 16px",
  borderRadius: "8px",
  color: "#9ca3af",
  textDecoration: "none",
  fontWeight: "500",
  transition: "all 0.2s ease-in-out",
  ":hover": {
    backgroundColor: "#1e293b",
    color: "#ffffff"
  }
};

export default App;

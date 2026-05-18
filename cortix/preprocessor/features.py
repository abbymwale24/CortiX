"""
CortiX Module 1 — Flow Feature Extraction

Extracts 16+ features from raw packets and aggregates them into
flow records over configurable time windows.
"""

import hashlib
import logging
import math
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("cortix.preprocessor.features")

try:
    from scapy.all import IP, TCP, UDP, ICMP, Raw
except ImportError:
    IP = TCP = UDP = ICMP = Raw = None

from cortix.config import config

# ──────────────────────────────────────────────
# Feature Names (canonical order)
# ──────────────────────────────────────────────
FLOW_FEATURE_NAMES = [
    "src_ip_hash",
    "dst_ip_hash",
    "src_port",
    "dst_port",
    "protocol",
    "packet_length",
    "inter_packet_interval",
    "flow_byte_count",
    "flow_packet_count",
    "flow_duration",
    "tcp_flags",
    "payload_entropy",
    "application_fingerprint",
    "time_of_day_bucket",
    "device_type_hint",
    "subnet_id",
]


def _shannon_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of a byte sequence."""
    if not data:
        return 0.0
    freq = defaultdict(int)
    for byte in data:
        freq[byte] += 1
    length = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _ip_to_hash(ip: str) -> float:
    """Hash an IP address to a deterministic float in [0, 1]."""
    if not ip:
        return 0.0
    h = int(hashlib.md5(ip.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF


def _tcp_flags_to_int(flags) -> int:
    """Convert Scapy TCP flags to integer bitmask."""
    if flags is None:
        return 0
    flag_map = {"F": 1, "S": 2, "R": 4, "P": 8, "A": 16, "U": 32, "E": 64, "C": 128}
    if isinstance(flags, int):
        return flags
    result = 0
    for ch in str(flags):
        result |= flag_map.get(ch, 0)
    return result


def _guess_app(port: int) -> int:
    """Port-based application fingerprint (simple mapping)."""
    app_map = {
        80: 1, 443: 2, 22: 3, 21: 4, 25: 5, 53: 6,
        110: 7, 143: 8, 3306: 9, 5432: 10, 8080: 11,
        3389: 12, 445: 13, 139: 14, 8443: 15,
    }
    return app_map.get(port, 0)


def _guess_device_type(ttl: int) -> int:
    """Infer OS/device type from TTL value."""
    if ttl <= 0:
        return 0
    if ttl <= 64:
        return 1  # Linux/macOS
    if ttl <= 128:
        return 2  # Windows
    return 3  # Network device / other


def extract_packet_features(packet) -> Optional[Dict[str, Any]]:
    """
    Extract features from a single Scapy packet.

    Returns:
        dict of feature values, or None if packet is not IP-based.
    """
    if IP is None or not packet.haslayer(IP):
        return None

    ip_layer = packet[IP]
    features = {
        "src_ip": ip_layer.src,
        "dst_ip": ip_layer.dst,
        "src_ip_hash": _ip_to_hash(ip_layer.src),
        "dst_ip_hash": _ip_to_hash(ip_layer.dst),
        "src_port": 0,
        "dst_port": 0,
        "protocol": ip_layer.proto,
        "packet_length": len(packet),
        "ttl": ip_layer.ttl,
        "tcp_flags": 0,
        "payload_entropy": 0.0,
        "timestamp": time.time(),
    }

    # TCP features
    if packet.haslayer(TCP):
        tcp = packet[TCP]
        features["src_port"] = tcp.sport
        features["dst_port"] = tcp.dport
        features["tcp_flags"] = _tcp_flags_to_int(tcp.flags)

    # UDP features
    elif packet.haslayer(UDP):
        udp = packet[UDP]
        features["src_port"] = udp.sport
        features["dst_port"] = udp.dport

    # Payload entropy
    if packet.haslayer(Raw):
        features["payload_entropy"] = _shannon_entropy(bytes(packet[Raw].load))

    # Derived features
    features["application_fingerprint"] = _guess_app(features["dst_port"])
    features["time_of_day_bucket"] = int(time.localtime().tm_hour)
    features["device_type_hint"] = _guess_device_type(features["ttl"])
    features["subnet_id"] = int(
        features["src_ip"].split(".")[2]
    ) if "." in features.get("src_ip", "") else 0

    return features


class FlowAggregator:
    """
    Aggregate individual packets into flow records over time windows.

    A flow is identified by the 5-tuple: (src_ip, dst_ip, src_port, dst_port, protocol).
    After each window, completed flows are emitted via callbacks.
    """

    def __init__(
        self,
        window_seconds: float = None,
        on_flow: Optional[Callable] = None,
    ):
        self.window = window_seconds or config.FLOW_WINDOW_SECONDS
        self._flows: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._callbacks: list[Callable] = []
        self._timer: Optional[threading.Timer] = None
        self._running = False

        if on_flow:
            self._callbacks.append(on_flow)

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def start(self):
        """Start the periodic flow emission timer."""
        self._running = True
        self._schedule_flush()
        logger.info("FlowAggregator started (window=%.1fs)", self.window)

    def stop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()
        self._flush()  # Emit remaining flows

    def add_packet(self, packet_features: Dict[str, Any]):
        """Add a packet's features to the current flow record."""
        if packet_features is None:
            return

        flow_key = (
            f"{packet_features.get('src_ip', '')}:"
            f"{packet_features.get('dst_ip', '')}:"
            f"{packet_features.get('src_port', 0)}:"
            f"{packet_features.get('dst_port', 0)}:"
            f"{packet_features.get('protocol', 0)}"
        )

        with self._lock:
            if flow_key not in self._flows:
                self._flows[flow_key] = {
                    **packet_features,
                    "flow_byte_count": 0,
                    "flow_packet_count": 0,
                    "flow_start_time": packet_features["timestamp"],
                    "flow_duration": 0.0,
                    "inter_packet_intervals": [],
                    "inter_packet_interval": 0.0,
                    "_last_packet_time": packet_features["timestamp"],
                }

            flow = self._flows[flow_key]
            flow["flow_byte_count"] += packet_features["packet_length"]
            flow["flow_packet_count"] += 1
            flow["flow_duration"] = (
                packet_features["timestamp"] - flow["flow_start_time"]
            )

            # Inter-packet interval
            ipi = packet_features["timestamp"] - flow["_last_packet_time"]
            flow["inter_packet_intervals"].append(ipi)
            flow["inter_packet_interval"] = np.mean(
                flow["inter_packet_intervals"][-50:]
            )
            flow["_last_packet_time"] = packet_features["timestamp"]

            # Update payload entropy (running max)
            flow["payload_entropy"] = max(
                flow.get("payload_entropy", 0),
                packet_features.get("payload_entropy", 0),
            )

    def _schedule_flush(self):
        if self._running:
            self._timer = threading.Timer(self.window, self._flush_and_reschedule)
            self._timer.daemon = True
            self._timer.start()

    def _flush_and_reschedule(self):
        self._flush()
        self._schedule_flush()

    def _flush(self):
        """Emit all current flows and reset."""
        with self._lock:
            flows = list(self._flows.values())
            self._flows.clear()

        for flow in flows:
            # Clean up internal fields
            flow.pop("inter_packet_intervals", None)
            flow.pop("_last_packet_time", None)
            flow.pop("flow_start_time", None)

            for cb in self._callbacks:
                try:
                    cb(flow)
                except Exception as exc:
                    logger.error("Flow callback error: %s", exc)

    def to_feature_vector(self, flow: Dict[str, Any]) -> np.ndarray:
        """
        Convert a flow record to a numeric feature vector.

        Returns:
            numpy array of shape (16,) with normalised features.
        """
        vec = np.array(
            [
                flow.get("src_ip_hash", 0.0),
                flow.get("dst_ip_hash", 0.0),
                flow.get("src_port", 0) / 65535.0,
                flow.get("dst_port", 0) / 65535.0,
                flow.get("protocol", 0) / 255.0,
                min(flow.get("packet_length", 0) / 1500.0, 1.0),
                min(flow.get("inter_packet_interval", 0) * 10.0, 1.0),
                min(flow.get("flow_byte_count", 0) / 1e6, 1.0),
                min(flow.get("flow_packet_count", 0) / 1000.0, 1.0),
                min(flow.get("flow_duration", 0) / 120.0, 1.0),
                flow.get("tcp_flags", 0) / 255.0,
                flow.get("payload_entropy", 0) / 8.0,
                flow.get("application_fingerprint", 0) / 15.0,
                flow.get("time_of_day_bucket", 0) / 23.0,
                flow.get("device_type_hint", 0) / 3.0,
                min(flow.get("subnet_id", 0) / 255.0, 1.0),
            ],
            dtype=np.float32,
        )
        return vec

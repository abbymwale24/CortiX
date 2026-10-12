"""
CortiX Module 1 — Live Packet Capture

Uses Scapy for real-time packet capture on a network interface.
Feeds raw packets into the feature extraction pipeline.
"""

import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger("cortix.preprocessor.capture")

try:
    from scapy.all import IP, TCP, UDP, AsyncSniffer, Raw, sniff
    from scapy.layers.l2 import Ether
    from scapy.utils import RawPcapReader
except ImportError:
    logger.warning("Scapy not installed — capture disabled")
    AsyncSniffer = None
    Ether = None
    RawPcapReader = None

from cortix.config import config


def _can_use_raw_socket() -> bool:
    """Test whether current process has permissions to create AF_PACKET raw sockets."""
    try:
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
        s.close()
        return True
    except (PermissionError, OSError):
        return False


class PacketCapture:
    """
    Live network packet capture engine.

    Runs an asynchronous Scapy sniffer on the configured interface
    and feeds packets to registered callbacks. Falls back to dumpcap streaming
    if raw sockets are not permitted for unprivileged users.
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        bpf_filter: Optional[str] = None,
    ):
        self.interface = interface or config.CAPTURE_INTERFACE
        self.bpf_filter = bpf_filter or config.BPF_FILTER or None
        self._sniffer: Optional[AsyncSniffer] = None
        self._dumpcap_proc: Optional[subprocess.Popen] = None
        self._dumpcap_thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable] = []
        self._packet_count = 0
        self._start_time = 0.0
        self._running = False

    def register_callback(self, callback: Callable):
        """Register a function to be called for each captured packet."""
        self._callbacks.append(callback)

    def start_capture(self):
        """Start live packet capture in a background thread."""
        if AsyncSniffer is None:
            logger.error("Scapy not available — cannot start capture")
            return

        self._running = True
        self._start_time = time.time()
        self._packet_count = 0

        # Check raw socket capability
        if _can_use_raw_socket():
            self._sniffer = AsyncSniffer(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._handle_packet,
                store=False,
            )
            self._sniffer.start()
            logger.info(
                "Capture started on %s via Scapy raw socket (filter: %s)",
                self.interface,
                self.bpf_filter or "none",
            )
            return

        # Fallback to dumpcap if available
        dumpcap_path = shutil.which("dumpcap")
        if dumpcap_path and RawPcapReader is not None and Ether is not None and self.interface:
            cmd: list[str] = [dumpcap_path, "-i", str(self.interface), "-q", "-w", "-"]
            if self.bpf_filter:
                cmd.extend(["-f", str(self.bpf_filter)])

            try:
                self._dumpcap_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                self._dumpcap_thread = threading.Thread(
                    target=self._stream_dumpcap, daemon=True
                )
                self._dumpcap_thread.start()
                logger.info(
                    "Capture started on %s via dumpcap stream (filter: %s)",
                    self.interface,
                    self.bpf_filter or "none",
                )
                return
            except Exception as exc:
                logger.error("Failed to start dumpcap capture: %s", exc)

        logger.error(
            "Cannot start capture on %s: raw socket permission denied and dumpcap unavailable",
            self.interface,
        )

    def _stream_dumpcap(self):
        """Read packets streamed from dumpcap stdout."""
        if self._dumpcap_proc is None or self._dumpcap_proc.stdout is None:
            return
        if RawPcapReader is None or Ether is None:
            return

        try:
            reader = RawPcapReader(self._dumpcap_proc.stdout)
            for pkt_data, _ in reader:
                if not self._running:
                    break
                try:
                    pkt = Ether(pkt_data)
                    self._handle_packet(pkt)
                except Exception as exc:
                    logger.debug("Error parsing packet from stream: %s", exc)
        except Exception as exc:
            if self._running:
                logger.debug("Dumpcap stream ended: %s", exc)

    def stop_capture(self):
        """Stop the packet capture."""
        self._running = False
        if self._sniffer:
            try:
                self._sniffer.stop()
            except Exception:
                pass
            self._sniffer = None

        if self._dumpcap_proc:
            try:
                self._dumpcap_proc.terminate()
                self._dumpcap_proc.wait(timeout=2)
            except Exception:
                pass
            self._dumpcap_proc = None

        if self._dumpcap_thread and self._dumpcap_thread.is_alive():
            self._dumpcap_thread.join(timeout=2)
            self._dumpcap_thread = None

        elapsed = time.time() - self._start_time
        logger.info(
            "Capture stopped — %d packets in %.1fs (%.0f pps)",
            self._packet_count,
            elapsed,
            self._packet_count / max(elapsed, 0.001),
        )

    def _handle_packet(self, packet):
        """Process a single captured packet through all callbacks."""
        self._packet_count += 1
        for cb in self._callbacks:
            try:
                cb(packet)
            except Exception as exc:
                logger.error("Callback error: %s", exc)

    @property
    def stats(self) -> dict:
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "packets_captured": self._packet_count,
            "elapsed_seconds": elapsed,
            "pps": self._packet_count / max(elapsed, 0.001),
            "interface": self.interface,
            "running": self._running,
        }


class OfflineCapture:
    """
    Read packets from PCAP files for offline analysis and testing.
    Uses the same callback interface as live capture.
    """

    def __init__(self, pcap_path: str):
        self.pcap_path = pcap_path
        self._callbacks: list[Callable] = []

    def register_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def replay(self, speed_factor: float = 1.0):
        """
        Replay packets from PCAP file.

        Args:
            speed_factor: 1.0 = real-time, 0 = as fast as possible
        """
        from scapy.all import PcapReader

        logger.info("Replaying PCAP: %s (speed: %.1fx)", self.pcap_path, speed_factor)
        count = 0
        prev_time = None

        with PcapReader(self.pcap_path) as reader:
            for packet in reader:
                if speed_factor > 0 and prev_time is not None:
                    delay = float(packet.time - prev_time) / speed_factor
                    if delay > 0:
                        time.sleep(delay)
                prev_time = packet.time

                for cb in self._callbacks:
                    try:
                        cb(packet)
                    except Exception as exc:
                        logger.error("Callback error: %s", exc)

                count += 1

        logger.info("Replay complete — %d packets", count)
        return count

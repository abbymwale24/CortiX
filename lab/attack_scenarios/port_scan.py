"""
CortiX Mininet Lab Scenario — Port Scan

Simulates active target port scanning using Scapy to generate high-rate syn packets.
"""

import sys
import argparse
import logging
import time

try:
    from scapy.all import IP, TCP, send
except ImportError:
    IP = TCP = send = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.lab.scenarios.port_scan")


def run_port_scan(target_ip: str, speed: str):
    if IP is None:
        logger.error("Scapy is required to run simulation.")
        return

    logger.info("Starting Port Scan Simulation targeting host: %s. Speed: %s", target_ip, speed)
    delay = 0.05 if speed == "fast" else 0.5
    
    # Scan ports 1 to 500
    for port in range(1, 501):
        try:
            # Craft raw SYN packet
            pkt = IP(dst=target_ip) / TCP(dport=port, flags="S")
            send(pkt, verbose=False)
            time.sleep(delay)
            if port % 50 == 0:
                logger.info("Sent %d probes...", port)
        except KeyboardInterrupt:
            logger.info("Scan cancelled.")
            break
            
    logger.info("Port Scan Simulation Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Port Scan")
    parser.add_argument("--target", type=str, default="10.0.0.2", help="Target victim IP")
    parser.add_argument("--rate", type=str, default="fast", choices=["fast", "slow"], help="Scan rate")
    args = parser.parse_args()

    run_port_scan(target_ip=args.target, speed=args.rate)

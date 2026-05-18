"""
CortiX Mininet Lab Scenario — SSH Brute Force

Simulates SSH credential brute forcing by executing high-rate TCP connection loops.
"""

import sys
import argparse
import logging
import time
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.lab.scenarios.brute_force")


def run_brute_force(target_ip: str, port: int, attempts: int):
    logger.info("Starting SSH Brute Force Simulation targeting host: %s on port %d", target_ip, port)
    
    success_count = 0
    fail_count = 0

    for i in range(1, attempts + 1):
        try:
            # Simulate a quick TCP connection loop typical of brute force scanners
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect((target_ip, port))
            # Send dummy SSH handshake payload
            s.sendall(b"SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.1\r\n")
            s.recv(1024)
            s.close()
            
            fail_count += 1
            if i % 10 == 0:
                logger.info("Executed %d login attempts...", i)
            time.sleep(0.1)
        except Exception:
            fail_count += 1
            time.sleep(0.05)
            
    logger.info("Brute Force Simulation Complete. Simulated %d attempts.", attempts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate SSH Brute Force")
    parser.add_argument("--target", type=str, default="10.0.0.2", help="Target victim IP")
    parser.add_argument("--port", type=int, default=22, help="SSH service port")
    parser.add_argument("--attempts", type=int, default=100, help="Number of attempts")
    args = parser.parse_args()

    run_brute_force(target_ip=args.target, port=args.port, attempts=args.attempts)

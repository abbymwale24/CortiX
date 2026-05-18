"""
CortiX Mininet Lab Scenario — Ransomware Simulation (SAFE)

Simulates safe mass-file encryption behavior solely within isolated decoy 
folders to trigger file system inotify/watchdog alerts.
"""

import os
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.lab.scenarios.ransomware_sim")


def run_ransomware_sim(target_dir: str):
    logger.info("Starting Ransomware Activity Simulation targeting directory: %s", target_dir)
    
    if not os.path.exists(target_dir):
        logger.info("Creating directory: %s", target_dir)
        os.makedirs(target_dir, exist_ok=True)

    # 1. Create decoy files
    logger.info("Populating decoy files...")
    for i in range(1, 60):
        filepath = os.path.join(target_dir, f"document_{i}.docx")
        with open(filepath, "w") as f:
            f.write("This is a highly valuable corporate asset file.")

    time.sleep(1)

    # 2. Simulate rapid encryption loop (overriding extensions)
    logger.info("Simulating rapid bulk file encryption loop...")
    count = 0
    start_time = time.time()
    
    for filename in os.listdir(target_dir):
        if filename.endswith(".docx"):
            filepath = os.path.join(target_dir, filename)
            new_filepath = os.path.join(target_dir, filename.replace(".docx", ".locked"))
            
            try:
                # Simulate read-then-rename typical of encryptors
                with open(filepath, "r") as f:
                    content = f.read()
                    
                # Safe rewrite
                with open(filepath, "w") as f:
                    f.write("[ENCRYPTED CONTENT MOCK] " + content)
                    
                os.rename(filepath, new_filepath)
                count += 1
                
                # Sleep briefly to mimic high rate but controlled pacing
                time.sleep(0.01)
            except Exception as exc:
                logger.error("Failed to encrypt decoy file: %s", exc)

    elapsed = time.time() - start_time
    logger.info("Ransomware simulation complete. Renamed/Encrypted %d files in %.2fs (%.1f files/s).", 
                count, elapsed, count / max(elapsed, 0.001))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Safe Ransomware Activity")
    parser.add_argument("--target-dir", type=str, default="/lab/decoy_files/", help="Decoy directory")
    args = parser.parse_args()

    run_ransomware_sim(target_dir=args.target_dir)

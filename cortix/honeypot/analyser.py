"""
CortiX Module 5 — Honeypot Capture & IOC Analyser

Extracts Indicators of Compromise (IOCs) such as C2 server IP addresses, 
suspicious domain beacons, encryption keys, and ransom notes from trap captures.
"""

import os
import re
import logging
from typing import Dict, List, Any

from cortix.config import config

logger = logging.getLogger("cortix.honeypot.analyser")


class HoneypotAnalyser:
    """
    IOC extraction post-analysis engine.
    
    Inspects text files, logs, or network PCAP outputs to extract C2,
    bitcoin wallet addresses, or onion domain markers.
    """

    def __init__(self):
        # Regex patterns for matching common ransom note contents
        self.ip_pattern = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
        self.btc_pattern = re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b")
        self.onion_pattern = re.compile(r"\b[a-z2-7]{16,56}\.onion\b")
        self.email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    def analyse_capture(self, log_file_path: str) -> Dict[str, Any]:
        """
        Scan a honeypot process log or ransom note file to extract threat indicators.
        
        Args:
            log_file_path: Path to plain text file containing captured telemetry
            
        Returns:
            dict containing lists of IPs, BTC addresses, onion domains, etc.
        """
        ioc_report = {
            "c2_ips": [],
            "bitcoin_addresses": [],
            "onion_domains": [],
            "contact_emails": [],
            "status": "NO_IOCS_FOUND",
        }

        if not os.path.exists(log_file_path):
            logger.warning("Target analysis file %s not found.", log_file_path)
            return ioc_report

        try:
            with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract distinct matches
            ips = list(set(self.ip_pattern.findall(content)))
            btcs = list(set(self.btc_pattern.findall(content)))
            onions = list(set(self.onion_pattern.findall(content)))
            emails = list(set(self.email_pattern.findall(content)))

            # Exclude standard local loopbacks from parsed IPs
            filtered_ips = [ip for ip in ips if not ip.startswith(("127.", "0.", "172.17."))]

            ioc_report["c2_ips"] = filtered_ips
            ioc_report["bitcoin_addresses"] = btcs
            ioc_report["onion_domains"] = onions
            ioc_report["contact_emails"] = emails
            
            if filtered_ips or btcs or onions or emails:
                ioc_report["status"] = "IOCS_EXTRACTED"
                logger.info("Successfully extracted %d IOCs from honeypot capture.", 
                            len(filtered_ips) + len(btcs) + len(onions) + len(emails))
            else:
                logger.info("No malware indicators extracted from %s.", log_file_path)

        except Exception as exc:
            logger.error("Failed to parse analysis log: %s", exc)
            ioc_report["status"] = "ANALYSIS_FAILED"

        return ioc_report

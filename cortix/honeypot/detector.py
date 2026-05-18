"""
CortiX Module 5 — Ransomware Behavior Detector

Tracks rename/write rates and matches YARA rules on modified files 
to detect potential ongoing ransomware execution patterns.
"""

import os
import time
import logging
from collections import deque

try:
    import yara
except ImportError:
    yara = None

from cortix.config import config

logger = logging.getLogger("cortix.honeypot.detector")

# Inline default YARA rule for matching ransomware patterns if external file missing
DEFAULT_YARA_RULES = """
rule RansomwareStrings {
    meta:
        description = "Detects common strings used in ransomware"
    strings:
        $vss = "vssadmin delete shadows" ascii nocase
        $vss2 = "Resize-Partition" ascii nocase
        $shadow = "shadowcopy" ascii nocase
        $note1 = "DECRYPT_INSTRUCTIONS" ascii nocase
        $note2 = "README_FOR_DECRYPT" ascii nocase
        $note3 = "YOUR_FILES_HAVE_BEEN_ENCRYPTED" ascii nocase
        $ext = ".locked" ascii
        $ext2 = ".crypted" ascii
    condition:
        any of them
}
"""


class RansomwareDetector:
    """
    Ransomware trigger evaluator.
    
    Analyses rates of file operations per second and runs YARA matches 
    against newly written/renamed files to catch encryption loops.
    """

    def __init__(self, rename_threshold: int = None):
        self.rename_threshold = rename_threshold or config.HONEYPOT_RENAME_THRESHOLD
        
        # History queue storing timestamps of rename actions
        self._rename_timestamps = deque()

        # Compiles YARA rules
        self.yara_rules = None
        self._load_yara()

    def _load_yara(self):
        if yara is None:
            logger.warning("yara-python not installed. YARA-based ransomware checks disabled.")
            return
            
        try:
            self.yara_rules = yara.compile(source=DEFAULT_YARA_RULES)
            logger.info("YARA signature matcher compiled successfully.")
        except Exception as exc:
            logger.error("Failed to compile YARA rules: %s", exc)

    def evaluate_activity(self, event_type: str, data: dict) -> dict:
        """
        Evaluate single filesystem activity event.
        
        Args:
            event_type: type of operation ('suspicious_rename', 'modified', etc.)
            data: event details containing paths, process IDs
            
        Returns:
            dict indicating whether a ransomware threat has been triggered.
        """
        now = time.time()
        
        if event_type == "suspicious_rename":
            self._rename_timestamps.append(now)
            
        # Remove timestamps older than 1 second
        while self._rename_timestamps and self._rename_timestamps[0] < now - 1.0:
            self._rename_timestamps.popleft()

        renames_per_sec = len(self._rename_timestamps)
        
        # 1. Check Rate Threshold Trigger
        rate_trigger = renames_per_sec >= self.rename_threshold

        # 2. Check YARA Signatures if file modified/created
        yara_trigger = False
        matched_rules = []
        
        target_path = data.get("dest_path") or data.get("src_path")
        if target_path and self.yara_rules and os.path.exists(target_path):
            try:
                # Scan first 10KB of file for speed
                with open(target_path, "rb") as f:
                    content = f.read(10240)
                matches = self.yara_rules.match(data=content)
                if matches:
                    yara_trigger = True
                    matched_rules = [m.rule for m in matches]
                    logger.warning("YARA signature MATCHED rule: %s on file: %s", matched_rules, target_path)
            except Exception:
                pass

        is_threat = rate_trigger or yara_trigger

        return {
            "is_ransomware": is_threat,
            "rate_trigger": rate_trigger,
            "yara_trigger": yara_trigger,
            "renames_per_second": renames_per_sec,
            "matched_rules": matched_rules,
            "target_path": target_path,
        }

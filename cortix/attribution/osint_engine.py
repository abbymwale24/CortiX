"""
CortiX Module 6 — Attacker Attribution Engine

Uses async requests to execute passive OSINT lookups (AbuseIPDB, VirusTotal, 
Shodan, Reverse DNS, WHOIS) to build a robust attacker attribution profile.
All calls run in parallel using asyncio to target ≤ 2 seconds resolution.
"""

import asyncio
import socket
import logging
import time
from typing import Dict, Any, List, Tuple

import aiohttp
try:
    import whois
except ImportError:
    whois = None

from cortix.config import config
from cortix.attribution.geoip import GeoIPResolver

logger = logging.getLogger("cortix.attribution.osint_engine")

# socket.gethostbyaddr and whois.whois are blocking calls with no built-in
# timeout of their own -- unlike the aiohttp calls below (which already cap
# at 1.5s), a slow/unresponsive resolver can leave them hanging for tens of
# seconds. Wrapping them in asyncio.wait_for bounds how long build_profile()
# waits on them; it does not forcibly kill the underlying OS call (a
# run_in_executor future isn't cancellable), so the thread-pool worker can
# still be tied up in the background, but it stops one bad lookup from
# stalling every other anomaly event's attribution.
_BLOCKING_LOOKUP_TIMEOUT_SEC = 5.0

# How long a completed profile is considered fresh before a new lookup is
# attempted for that IP.
_DEFAULT_CACHE_TTL_SEC = 600.0


class AttackerAttributionEngine:
    """
    Attribution pipeline running concurrent, read-only queries against OSINT endpoints.

    main.py triggers a new attribution lookup on every single anomaly event,
    with no deduplication of its own. Without caching, a handful of hosts
    bouncing repeated anomalies back and forth (the common case for a noisy
    live demo) fans out into hundreds of overlapping whois/geoip/AbuseIPDB/
    VirusTotal/Shodan lookups for the *same* IPs, which is both wasteful and
    can exhaust the process's outbound sockets (observed as whois connection
    timeouts under load). This engine instead:

    - caches a completed profile per IP for `cache_ttl_sec`, returning it
      immediately with no network calls while still fresh
    - coalesces concurrent requests for the same IP onto a single in-flight
      lookup, so a burst of anomalies for one attacker triggers one lookup,
      not one per event
    """

    def __init__(self, cache_ttl_sec: float = _DEFAULT_CACHE_TTL_SEC):
        self.geoip = GeoIPResolver()
        self.cache_ttl_sec = cache_ttl_sec
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._in_flight: Dict[str, "asyncio.Future[Dict[str, Any]]"] = {}

    async def build_profile(self, src_ip: str) -> Dict[str, Any]:
        """
        Returns an attribution profile for a target IP address.

        Serves a cached profile immediately if one younger than
        `cache_ttl_sec` exists. If a lookup for this IP is already running,
        awaits that same lookup rather than starting a duplicate one.
        Otherwise performs a fresh lookup and caches the result.
        """
        now = time.monotonic()

        cached = self._cache.get(src_ip)
        if cached is not None:
            cached_at, profile = cached
            if now - cached_at < self.cache_ttl_sec:
                return profile

        in_flight = self._in_flight.get(src_ip)
        if in_flight is not None:
            return await in_flight

        loop = asyncio.get_event_loop()
        future: "asyncio.Future[Dict[str, Any]]" = loop.create_future()
        self._in_flight[src_ip] = future
        try:
            profile = await self._build_profile_uncached(src_ip)
        except Exception as exc:
            future.set_exception(exc)
            # Mark it retrieved immediately: this caller re-raises below, and
            # if no concurrent caller is awaiting `future`, asyncio otherwise
            # logs a spurious "exception was never retrieved" warning once
            # the future is garbage collected.
            future.exception()
            self._in_flight.pop(src_ip, None)
            raise
        else:
            future.set_result(profile)
            self._cache[src_ip] = (now, profile)
            self._in_flight.pop(src_ip, None)
            return profile

    async def _build_profile_uncached(self, src_ip: str) -> Dict[str, Any]:
        """
        Builds a comprehensive profile for a target IP address in under 2 seconds.
        """
        t0 = time.perf_counter()
        
        # 1. Geolocation lookup (Offline - fast)
        geo_data = self.geoip.resolve(src_ip)

        # Skip OSINT api calls for private subnet addresses
        if src_ip.startswith(("127.", "192.168.", "10.", "172.16.", "172.17.")):
            profile = {
                "ip": src_ip,
                **geo_data,
                "hostname": "local.domain",
                "abuse_score": 0,
                "known_malicious": False,
                "attack_categories": ["internal"],
                "vt_malicious": 0,
                "shodan_ports": [],
                "threat_level": "LOW",
                "attribution_duration_sec": time.perf_counter() - t0,
            }
            return profile

        # Gather remaining details concurrently
        tasks = [
            self._resolve_rdns(src_ip),
            self._query_abuseipdb(src_ip),
            self._query_virustotal(src_ip),
            self._query_shodan(src_ip),
            self._query_whois(src_ip),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Parse gathered details
        hostname = results[0] if not isinstance(results[0], Exception) else "Unknown"
        abuse_score, categories = results[1] if not isinstance(results[1], Exception) else (0, [])
        vt_malicious = results[2] if not isinstance(results[2], Exception) else 0
        shodan_ports = results[3] if not isinstance(results[3], Exception) else []
        whois_domain = results[4] if not isinstance(results[4], Exception) else "Unknown"

        # Determine overall threat level
        threat_level = "LOW"
        if abuse_score > 75 or vt_malicious > 10:
            threat_level = "CRITICAL"
        elif abuse_score > 40 or vt_malicious > 2:
            threat_level = "HIGH"
        elif abuse_score > 15:
            threat_level = "MEDIUM"

        profile = {
            "ip": src_ip,
            **geo_data,
            "hostname": hostname,
            "abuse_score": int(abuse_score),
            "known_malicious": abuse_score > 20 or vt_malicious > 0,
            "attack_categories": categories,
            "vt_malicious": int(vt_malicious),
            "shodan_ports": shodan_ports,
            "whois_domain": whois_domain,
            "threat_level": threat_level,
            "attribution_duration_sec": float(time.perf_counter() - t0),
        }

        logger.info("Attribution profile compiled for %s in %.2fs. Level: %s", 
                    src_ip, profile["attribution_duration_sec"], threat_level)
        return profile

    async def _resolve_rdns(self, ip: str) -> str:
        """Resolve IP address to reverse DNS hostname."""
        try:
            loop = asyncio.get_event_loop()
            # Run blocking socket lookup inside default executor thread pool,
            # bounded so an unresponsive resolver can't stall build_profile().
            res = await asyncio.wait_for(
                loop.run_in_executor(None, socket.gethostbyaddr, ip),
                timeout=_BLOCKING_LOOKUP_TIMEOUT_SEC,
            )
            return res[0]
        except Exception:
            return "Unknown"

    async def _query_abuseipdb(self, ip: str) -> tuple[int, List[str]]:
        """Query AbuseIPDB API."""
        key = config.ABUSEIPDB_API_KEY
        if not key or key == "YOUR_KEY":
            return 0, []

        url = f"https://api.abuseipdb.com/api/v2/check?ipAddress={ip}&maxAgeInDays=90"
        headers = {"Key": key, "Accept": "application/json"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=1.5) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        data = res.get("data", {})
                        score = data.get("abuseConfidenceScore", 0)
                        
                        # Extract categories
                        reports = data.get("reports", [])
                        categories = []
                        for r in reports:
                            categories.extend(r.get("categories", []))
                        # Deduplicate
                        categories = list(set(categories))
                        return score, categories
        except Exception:
            pass
        return 0, []

    async def _query_virustotal(self, ip: str) -> int:
        """Query VirusTotal IP endpoints."""
        key = config.VIRUSTOTAL_API_KEY
        if not key or key == "YOUR_KEY":
            return 0

        url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
        headers = {"x-apikey": key}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=1.5) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        stats = res.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                        return stats.get("malicious", 0)
        except Exception:
            pass
        return 0

    async def _query_shodan(self, ip: str) -> List[int]:
        """Query Shodan Host lookup endpoints."""
        key = config.SHODAN_API_KEY
        if not key or key == "YOUR_KEY":
            return []

        url = f"https://api.shodan.io/shodan/host/{ip}?key={key}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=1.5) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        return res.get("ports", [])
        except Exception:
            pass
        return []

    async def _query_whois(self, ip: str) -> str:
        """Query whois details."""
        if whois is None:
            return "Unknown"

        try:
            loop = asyncio.get_event_loop()
            # Run blocking WHOIS request inside executor thread pool, bounded
            # so an unresponsive WHOIS server can't stall build_profile().
            w = await asyncio.wait_for(
                loop.run_in_executor(None, whois.whois, ip),
                timeout=_BLOCKING_LOOKUP_TIMEOUT_SEC,
            )
            return w.get("domain_name") or w.get("registrar") or "Unknown"
        except Exception:
            return "Unknown"

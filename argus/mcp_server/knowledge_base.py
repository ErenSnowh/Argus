"""
knowledge_base.py
-------------------
Integration with open cybersecurity knowledge bases:
  - MITRE ATT&CK Enterprise (via STIX/TAXII from GitHub)
  - CAPEC (Common Attack Pattern Enumeration and Classification)
  - CVE/NVD (Common Vulnerabilities and Exposures / National Vulnerability Database)

All data is fetched from public sources and cached locally for offline use.
This module provides enrichment context for ARGUS detections by linking
flow-level classifications to known attack patterns, techniques, and
vulnerability identifiers.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# Cache directory for downloaded knowledge bases
CACHE_DIR = Path(__file__).parent.parent / "data" / "knowledge_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# MITRE ATT&CK Enterprise STIX bundle URL (GitHub mirror — always available)
ATTACK_STIX_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
ATTACK_CACHE_FILE = CACHE_DIR / "enterprise_attack.json"

# CAPEC list URL
CAPEC_URL = "https://raw.githubusercontent.com/mitre/cti/master/capec/2.1/stix-capec.json"
CAPEC_CACHE_FILE = CACHE_DIR / "capec.json"

# NVD API (public, rate-limited to 5 requests/30s without API key)
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY = os.environ.get("NVD_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttackTechnique:
    """A MITRE ATT&CK technique with full metadata."""
    technique_id: str
    name: str
    tactic: str
    description: str
    platforms: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)
    detection: str = ""
    mitigations: list[str] = field(default_factory=list)
    url: str = ""


@dataclass
class CAPECPattern:
    """A CAPEC attack pattern."""
    capec_id: str
    name: str
    description: str
    severity: str = ""
    likelihood: str = ""
    related_cwe: list[str] = field(default_factory=list)
    related_attack: list[str] = field(default_factory=list)


@dataclass
class CVEEntry:
    """A CVE vulnerability entry."""
    cve_id: str
    description: str
    severity: str = ""
    cvss_score: float = 0.0
    published: str = ""
    references: list[str] = field(default_factory=list)
    affected_products: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MITRE ATT&CK knowledge base
# ---------------------------------------------------------------------------

class MITREAttackKB:
    """MITRE ATT&CK Enterprise knowledge base.

    Downloads the full STIX bundle from the MITRE GitHub repository and
    provides lookup methods for techniques, tactics, and mitigations.
    Data is cached locally for offline use.
    """

    def __init__(self):
        self._techniques: dict[str, AttackTechnique] = {}
        self._tactics: dict[str, list[str]] = {}  # tactic → [technique_ids]
        self._loaded = False

    def _download_if_needed(self):
        """Download the ATT&CK STIX bundle if not cached or stale."""
        if ATTACK_CACHE_FILE.exists():
            # Use cache if less than 7 days old
            age = time.time() - ATTACK_CACHE_FILE.stat().st_mtime
            if age < 7 * 24 * 3600:
                return

        try:
            print("Downloading MITRE ATT&CK Enterprise STIX bundle...")
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(ATTACK_STIX_URL)
                resp.raise_for_status()
                ATTACK_CACHE_FILE.write_bytes(resp.content)
                print(f"  Saved to {ATTACK_CACHE_FILE} ({len(resp.content) / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"  Warning: could not download ATT&CK data: {e}")

    def load(self):
        """Load and parse the ATT&CK STIX bundle."""
        if self._loaded:
            return

        self._download_if_needed()

        if not ATTACK_CACHE_FILE.exists():
            print("ATT&CK data not available. Using built-in mappings only.")
            self._loaded = True
            return

        try:
            data = json.loads(ATTACK_CACHE_FILE.read_text(encoding="utf-8"))
            objects = data.get("objects", [])

            # Build tactic name lookup
            tactic_names = {}
            for obj in objects:
                if obj.get("type") == "x-mitre-tactic":
                    short_name = obj.get("x_mitre_shortname", "")
                    name = obj.get("name", "")
                    tactic_names[short_name] = name

            # Parse techniques
            for obj in objects:
                if obj.get("type") != "attack-pattern":
                    continue
                if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
                    continue

                # Extract technique ID
                ext_refs = obj.get("external_references", [])
                tech_id = ""
                url = ""
                for ref in ext_refs:
                    if ref.get("source_name") == "mitre-attack":
                        tech_id = ref.get("external_id", "")
                        url = ref.get("url", "")
                        break

                if not tech_id:
                    continue

                # Extract tactic
                kill_chain = obj.get("kill_chain_phases", [])
                tactic = ""
                for phase in kill_chain:
                    if phase.get("kill_chain_name") == "mitre-attack":
                        tactic = tactic_names.get(phase.get("phase_name", ""), phase.get("phase_name", ""))
                        break

                # Extract platforms
                platforms = obj.get("x_mitre_platforms", [])

                # Extract data sources
                data_sources = obj.get("x_mitre_data_sources", [])

                # Extract detection
                detection = obj.get("x_mitre_detection", "")

                technique = AttackTechnique(
                    technique_id=tech_id,
                    name=obj.get("name", ""),
                    tactic=tactic,
                    description=obj.get("description", "")[:500],
                    platforms=platforms,
                    data_sources=data_sources[:5],
                    detection=detection[:300] if detection else "",
                    url=url,
                )

                self._techniques[tech_id] = technique

                if tactic:
                    self._tactics.setdefault(tactic, []).append(tech_id)

            # Parse mitigations and link to techniques
            mitigation_names = {}
            for obj in objects:
                if obj.get("type") == "course-of-action":
                    mid = ""
                    for ref in obj.get("external_references", []):
                        if ref.get("source_name") == "mitre-attack":
                            mid = ref.get("external_id", "")
                    if mid:
                        mitigation_names[obj["id"]] = obj.get("name", mid)

            for obj in objects:
                if obj.get("type") == "relationship" and obj.get("relationship_type") == "mitigates":
                    src = obj.get("source_ref", "")
                    tgt = obj.get("target_ref", "")
                    if src in mitigation_names:
                        # Find the technique
                        for tech in self._techniques.values():
                            if tgt.endswith(tech.technique_id.lower().replace(".", "-")):
                                tech.mitigations.append(mitigation_names[src])

            self._loaded = True
            print(f"  Loaded {len(self._techniques)} ATT&CK techniques across {len(self._tactics)} tactics")

        except Exception as e:
            print(f"  Error parsing ATT&CK data: {e}")
            self._loaded = True

    def lookup_technique(self, technique_id: str) -> AttackTechnique | None:
        """Look up a technique by ID (e.g., T1498, T1595.001)."""
        self.load()
        return self._techniques.get(technique_id)

    def search_techniques(self, query: str, max_results: int = 10) -> list[AttackTechnique]:
        """Search techniques by name or description."""
        self.load()
        query_lower = query.lower()
        results = []
        for tech in self._techniques.values():
            if query_lower in tech.name.lower() or query_lower in tech.description.lower():
                results.append(tech)
            if len(results) >= max_results:
                break
        return results

    def get_tactic_techniques(self, tactic: str) -> list[AttackTechnique]:
        """Get all techniques for a given tactic."""
        self.load()
        tech_ids = self._tactics.get(tactic, [])
        return [self._techniques[tid] for tid in tech_ids if tid in self._techniques]

    def get_stats(self) -> dict:
        self.load()
        return {
            "techniques_loaded": len(self._techniques),
            "tactics": list(self._tactics.keys()),
            "cache_file": str(ATTACK_CACHE_FILE),
            "cache_exists": ATTACK_CACHE_FILE.exists(),
        }


# ---------------------------------------------------------------------------
# CAPEC knowledge base
# ---------------------------------------------------------------------------

class CAPECKB:
    """CAPEC (Common Attack Pattern Enumeration) knowledge base.

    Provides lookup for attack patterns that describe HOW attacks are
    carried out, linking to CWE weaknesses and ATT&CK techniques.
    """

    def __init__(self):
        self._patterns: dict[str, CAPECPattern] = {}
        self._loaded = False

    def _download_if_needed(self):
        if CAPEC_CACHE_FILE.exists():
            age = time.time() - CAPEC_CACHE_FILE.stat().st_mtime
            if age < 7 * 24 * 3600:
                return
        try:
            print("Downloading CAPEC STIX bundle...")
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(CAPEC_URL)
                resp.raise_for_status()
                CAPEC_CACHE_FILE.write_bytes(resp.content)
                print(f"  Saved to {CAPEC_CACHE_FILE}")
        except Exception as e:
            print(f"  Warning: could not download CAPEC data: {e}")

    def load(self):
        if self._loaded:
            return

        self._download_if_needed()
        if not CAPEC_CACHE_FILE.exists():
            self._loaded = True
            return

        try:
            data = json.loads(CAPEC_CACHE_FILE.read_text(encoding="utf-8"))
            for obj in data.get("objects", []):
                if obj.get("type") != "attack-pattern":
                    continue

                capec_id = ""
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "capec":
                        capec_id = ref.get("external_id", "")
                        break

                if not capec_id:
                    continue

                # Extract related CWE and ATT&CK
                related_cwe = []
                related_attack = []
                for ref in obj.get("external_references", []):
                    if ref.get("source_name") == "cwe":
                        related_cwe.append(ref.get("external_id", ""))
                    elif ref.get("source_name") == "mitre-attack":
                        related_attack.append(ref.get("external_id", ""))

                severity = ""
                likelihood = ""
                for ext in obj.get("x_capec_likelihood_of_attack", ""):
                    likelihood = ext if isinstance(ext, str) else ""
                if isinstance(obj.get("x_capec_typical_severity"), str):
                    severity = obj["x_capec_typical_severity"]

                pattern = CAPECPattern(
                    capec_id=capec_id,
                    name=obj.get("name", ""),
                    description=obj.get("description", "")[:400],
                    severity=severity,
                    likelihood=likelihood,
                    related_cwe=related_cwe,
                    related_attack=related_attack,
                )
                self._patterns[capec_id] = pattern

            self._loaded = True
            print(f"  Loaded {len(self._patterns)} CAPEC attack patterns")

        except Exception as e:
            print(f"  Error parsing CAPEC data: {e}")
            self._loaded = True

    def lookup_pattern(self, capec_id: str) -> CAPECPattern | None:
        self.load()
        return self._patterns.get(capec_id)

    def search_patterns(self, query: str, max_results: int = 10) -> list[CAPECPattern]:
        self.load()
        query_lower = query.lower()
        results = []
        for pattern in self._patterns.values():
            if query_lower in pattern.name.lower() or query_lower in pattern.description.lower():
                results.append(pattern)
            if len(results) >= max_results:
                break
        return results

    def find_patterns_for_technique(self, technique_id: str) -> list[CAPECPattern]:
        """Find CAPEC patterns related to a MITRE ATT&CK technique."""
        self.load()
        return [p for p in self._patterns.values() if technique_id in p.related_attack]


# ---------------------------------------------------------------------------
# CVE/NVD knowledge base
# ---------------------------------------------------------------------------

class CVENVDKB:
    """CVE/NVD (National Vulnerability Database) knowledge base.

    Queries the NVD API for vulnerability information. Results are cached
    locally. Respects NVD rate limits (5 requests per 30 seconds without
    API key, 50 with API key).
    """

    def __init__(self):
        self._cache: dict[str, CVEEntry] = {}
        self._cache_file = CACHE_DIR / "cve_cache.json"
        self._load_cache()
        self._last_request_time = 0.0

    def _load_cache(self):
        if self._cache_file.exists():
            try:
                data = json.loads(self._cache_file.read_text())
                for cve_id, entry in data.items():
                    self._cache[cve_id] = CVEEntry(**entry)
            except Exception:
                pass

    def _save_cache(self):
        data = {}
        for cve_id, entry in self._cache.items():
            data[cve_id] = {
                "cve_id": entry.cve_id,
                "description": entry.description,
                "severity": entry.severity,
                "cvss_score": entry.cvss_score,
                "published": entry.published,
                "references": entry.references[:5],
                "affected_products": entry.affected_products[:5],
            }
        self._cache_file.write_text(json.dumps(data, indent=2))

    def _rate_limit(self):
        """Respect NVD rate limits."""
        min_interval = 6.0 if not NVD_API_KEY else 0.6
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def lookup_cve(self, cve_id: str) -> CVEEntry | None:
        """Look up a specific CVE by ID (e.g., CVE-2021-44228)."""
        cve_id = cve_id.upper().strip()

        # Check cache first
        if cve_id in self._cache:
            return self._cache[cve_id]

        # Query NVD API
        try:
            self._rate_limit()
            headers = {}
            if NVD_API_KEY:
                headers["apiKey"] = NVD_API_KEY

            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    NVD_API_URL,
                    params={"cveId": cve_id},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            vulns = data.get("vulnerabilities", [])
            if not vulns:
                return None

            cve_data = vulns[0].get("cve", {})

            # Extract description
            descriptions = cve_data.get("descriptions", [])
            desc = ""
            for d in descriptions:
                if d.get("lang") == "en":
                    desc = d.get("value", "")
                    break

            # Extract CVSS score
            metrics = cve_data.get("metrics", {})
            cvss_score = 0.0
            severity = ""
            for version in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                if version in metrics:
                    m = metrics[version][0]
                    cvss_data = m.get("cvssData", {})
                    cvss_score = cvss_data.get("baseScore", 0.0)
                    severity = cvss_data.get("baseSeverity", m.get("baseSeverity", ""))
                    break

            # Extract references
            refs = [r.get("url", "") for r in cve_data.get("references", [])[:5]]

            # Extract affected products
            products = []
            for config in cve_data.get("configurations", []):
                for node in config.get("nodes", []):
                    for match in node.get("cpeMatch", []):
                        criteria = match.get("criteria", "")
                        if criteria:
                            parts = criteria.split(":")
                            if len(parts) >= 5:
                                products.append(f"{parts[3]}:{parts[4]}")

            entry = CVEEntry(
                cve_id=cve_id,
                description=desc[:500],
                severity=severity,
                cvss_score=cvss_score,
                published=cve_data.get("published", ""),
                references=refs,
                affected_products=products[:5],
            )

            self._cache[cve_id] = entry
            self._save_cache()
            return entry

        except Exception as e:
            print(f"Warning: CVE lookup failed for {cve_id}: {e}")
            return None

    def search_cves(self, keyword: str, max_results: int = 5) -> list[CVEEntry]:
        """Search NVD for CVEs matching a keyword."""
        try:
            self._rate_limit()
            headers = {}
            if NVD_API_KEY:
                headers["apiKey"] = NVD_API_KEY

            with httpx.Client(timeout=15.0) as client:
                resp = client.get(
                    NVD_API_URL,
                    params={
                        "keywordSearch": keyword,
                        "resultsPerPage": max_results,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()

            results = []
            for vuln in data.get("vulnerabilities", []):
                cve_data = vuln.get("cve", {})
                cve_id = cve_data.get("id", "")

                descriptions = cve_data.get("descriptions", [])
                desc = ""
                for d in descriptions:
                    if d.get("lang") == "en":
                        desc = d.get("value", "")
                        break

                metrics = cve_data.get("metrics", {})
                cvss_score = 0.0
                severity = ""
                for version in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                    if version in metrics:
                        m = metrics[version][0]
                        cvss_data = m.get("cvssData", {})
                        cvss_score = cvss_data.get("baseScore", 0.0)
                        severity = cvss_data.get("baseSeverity", m.get("baseSeverity", ""))
                        break

                entry = CVEEntry(
                    cve_id=cve_id,
                    description=desc[:300],
                    severity=severity,
                    cvss_score=cvss_score,
                    published=cve_data.get("published", ""),
                )
                results.append(entry)
                self._cache[cve_id] = entry

            if results:
                self._save_cache()
            return results

        except Exception as e:
            print(f"Warning: CVE search failed for '{keyword}': {e}")
            return []


# ---------------------------------------------------------------------------
# Technique-to-label mapping for enrichment
# ---------------------------------------------------------------------------

# Maps our detection labels to relevant ATT&CK technique IDs for deep lookup
LABEL_TO_TECHNIQUES = {
    "DDoS": ["T1498", "T1498.001", "T1498.002", "T1499"],
    "PortScan": ["T1595", "T1595.001", "T1595.002", "T1046"],
    "BruteForce": ["T1110", "T1110.001", "T1110.002", "T1110.003", "T1110.004"],
    "WebAttack": ["T1190", "T1059.007", "T1203"],
    "LateralMovement": ["T1021", "T1021.001", "T1021.002", "T1021.004", "T1570"],
    "Botnet": ["T1071", "T1071.001", "T1573", "T1095"],
    "Exfiltration": ["T1041", "T1048", "T1567", "T1029"],
}

# Maps labels to relevant CAPEC patterns (by keyword)
LABEL_TO_CAPEC_KEYWORDS = {
    "DDoS": "denial of service",
    "PortScan": "port scanning",
    "BruteForce": "brute force",
    "WebAttack": "injection",
    "LateralMovement": "lateral movement",
    "Botnet": "command and control",
    "Exfiltration": "exfiltration",
}

# Maps labels to CVE search keywords
LABEL_TO_CVE_KEYWORDS = {
    "WebAttack": "SQL injection remote code execution",
    "BruteForce": "authentication bypass",
    "Botnet": "command and control backdoor",
    "Exfiltration": "data exfiltration",
}


# ---------------------------------------------------------------------------
# Module-level singletons (lazy initialization)
# ---------------------------------------------------------------------------

_attack_kb: MITREAttackKB | None = None
_capec_kb: CAPECKB | None = None
_cve_kb: CVENVDKB | None = None


def get_attack_kb() -> MITREAttackKB:
    global _attack_kb
    if _attack_kb is None:
        _attack_kb = MITREAttackKB()
    return _attack_kb


def get_capec_kb() -> CAPECKB:
    global _capec_kb
    if _capec_kb is None:
        _capec_kb = CAPECKB()
    return _capec_kb


def get_cve_kb() -> CVENVDKB:
    global _cve_kb
    if _cve_kb is None:
        _cve_kb = CVENVDKB()
    return _cve_kb


# ---------------------------------------------------------------------------
# Public API: unified enrichment
# ---------------------------------------------------------------------------

def enrich_detection_with_kb(
    label: str,
    technique_id: str | None = None,
) -> dict:
    """Enrich a detection label with knowledge from ATT&CK, CAPEC, and CVE/NVD.

    This is the main function exposed as an MCP tool. Given a detection label
    (and optionally a specific technique ID), it returns enriched context from
    all three knowledge bases.
    """
    result = {
        "label": label,
        "attack_techniques": [],
        "capec_patterns": [],
        "related_cves": [],
    }

    # 1. MITRE ATT&CK enrichment
    attack_kb = get_attack_kb()
    technique_ids = LABEL_TO_TECHNIQUES.get(label, [])
    if technique_id:
        technique_ids = [technique_id] + [t for t in technique_ids if t != technique_id]

    for tid in technique_ids[:5]:
        tech = attack_kb.lookup_technique(tid)
        if tech:
            result["attack_techniques"].append({
                "technique_id": tech.technique_id,
                "name": tech.name,
                "tactic": tech.tactic,
                "description": tech.description[:200],
                "platforms": tech.platforms,
                "data_sources": tech.data_sources[:3],
                "detection": tech.detection[:150],
                "mitigations": tech.mitigations[:3],
                "url": tech.url,
            })

    # 2. CAPEC enrichment
    capec_kb = get_capec_kb()
    capec_keyword = LABEL_TO_CAPEC_KEYWORDS.get(label)
    if capec_keyword:
        patterns = capec_kb.search_patterns(capec_keyword, max_results=3)
        for p in patterns:
            result["capec_patterns"].append({
                "capec_id": p.capec_id,
                "name": p.name,
                "description": p.description[:200],
                "severity": p.severity,
                "related_cwe": p.related_cwe[:3],
                "related_attack": p.related_attack[:3],
            })

    # 3. CVE/NVD enrichment (only for applicable attack types)
    cve_keyword = LABEL_TO_CVE_KEYWORDS.get(label)
    if cve_keyword:
        cve_kb = get_cve_kb()
        cves = cve_kb.search_cves(cve_keyword, max_results=3)
        for cve in cves:
            result["related_cves"].append({
                "cve_id": cve.cve_id,
                "description": cve.description[:200],
                "severity": cve.severity,
                "cvss_score": cve.cvss_score,
                "published": cve.published,
            })

    return result


def lookup_attack_technique(technique_id: str) -> dict | None:
    """Look up a specific ATT&CK technique by ID."""
    kb = get_attack_kb()
    tech = kb.lookup_technique(technique_id)
    if not tech:
        return None
    return {
        "technique_id": tech.technique_id,
        "name": tech.name,
        "tactic": tech.tactic,
        "description": tech.description,
        "platforms": tech.platforms,
        "data_sources": tech.data_sources,
        "detection": tech.detection,
        "mitigations": tech.mitigations,
        "url": tech.url,
    }


def lookup_cve(cve_id: str) -> dict | None:
    """Look up a specific CVE by ID."""
    kb = get_cve_kb()
    entry = kb.lookup_cve(cve_id)
    if not entry:
        return None
    return {
        "cve_id": entry.cve_id,
        "description": entry.description,
        "severity": entry.severity,
        "cvss_score": entry.cvss_score,
        "published": entry.published,
        "references": entry.references,
        "affected_products": entry.affected_products,
    }


def get_kb_status() -> dict:
    """Return the status of all knowledge bases."""
    cve_cache_file = CACHE_DIR / "cve_cache.json"
    cve_cache_exists = cve_cache_file.exists()
    n_cached_cves = 0
    if cve_cache_exists:
        try:
            n_cached_cves = len(json.loads(cve_cache_file.read_text()))
        except Exception:
            pass

    return {
        "attack": {
            "cache_exists": ATTACK_CACHE_FILE.exists(),
            "cache_path": str(ATTACK_CACHE_FILE),
            "source_url": ATTACK_STIX_URL,
        },
        "capec": {
            "cache_exists": CAPEC_CACHE_FILE.exists(),
            "cache_path": str(CAPEC_CACHE_FILE),
            "source_url": CAPEC_URL,
        },
        "cve_nvd": {
            "cache_exists": cve_cache_exists,
            "cached_entries": n_cached_cves,
            "api_url": NVD_API_URL,
            "api_key_set": bool(NVD_API_KEY),
            "cache_path": str(cve_cache_file),
        },
    }


if __name__ == "__main__":
    print("ARGUS Knowledge Base Status")
    print("=" * 60)
    status = get_kb_status()
    for kb_name, info in status.items():
        print(f"\n  {kb_name}:")
        for k, v in info.items():
            print(f"    {k}: {v}")

    # Demo: enrich a DDoS detection
    print("\n\nDemo: Enriching 'DDoS' detection...")
    result = enrich_detection_with_kb("DDoS")
    print(f"  ATT&CK techniques: {len(result['attack_techniques'])}")
    print(f"  CAPEC patterns: {len(result['capec_patterns'])}")
    print(f"  Related CVEs: {len(result['related_cves'])}")

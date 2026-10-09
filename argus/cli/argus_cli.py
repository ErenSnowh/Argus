#!/usr/bin/env python3
"""
argus_cli.py
-------------
ARGUS Agent Skills CLI -- each subcommand is a discrete, independently
invokable "agent skill" that wraps one stage of the SOC pipeline, so an
analyst (or another automation, e.g. a SOAR playbook or cron job) can call
exactly the capability they need without spinning up the full multi-agent
session. Each subcommand maps to one stage of the SIH 2026 PS-26153
forecasting-and-response pipeline. (The multi-skill CLI pattern began as a
Google ADK "agent skills" artifact and was extended for this problem statement.)

The Gemini API key is auto-loaded — no manual setup needed.

Install (editable, from repo root):
    pip install -e .

Usage:
    argus train                                   # train the RF detector
    argus detect --flow flow.json                 # classify a flow
    argus detect --random                         # classify a random demo flow
    argus enrich 185.220.101.45                    # IOC threat-intel lookup
    argus forensics data/sample_portscan.pcap      # PCAP summary
    argus attack DDoS                              # MITRE ATT&CK lookup
    argus playbook DDoS --confidence 0.95          # propose remediation
    argus investigate --random --pcap data/sample_portscan.pcap --ip 185.220.101.45
                                                    # full offline pipeline
    argus investigate --random --live               # full LIVE Gemini pipeline
    argus audit verify                             # check audit-log integrity
    argus train-world-model                        # train the World Model Transformer
    argus forecast --random --steps 5              # K-step infiltration forecast
    argus benchmark                                # World Model vs LR baseline
    argus datasets                                 # list supported datasets
    argus kb-enrich DDoS                           # enrich with ATT&CK/CAPEC/CVE
    argus kb-status                                # knowledge base cache status
    argus lookup-cve CVE-2021-44228                # look up a specific CVE
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: F401, E402 -- auto-sets GOOGLE_API_KEY


def cmd_train(args: argparse.Namespace) -> None:
    from ml.model import train

    metrics = train(n_per_class=args.n_per_class, tune=not args.fast)
    print(f"Accuracy: {metrics['accuracy']:.4f}  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Model + metrics written to ml/artifacts/")


def _random_demo_flow() -> dict:
    from ml.flow_features import generate_dataset, FEATURE_COLUMNS

    df = generate_dataset(n_per_class=5, seed=random.randint(0, 99999))
    row = df.sample(1).iloc[0]
    return {col: float(row[col]) for col in FEATURE_COLUMNS}


def cmd_detect(args: argparse.Namespace) -> None:
    from ml.model import FlowClassifier

    if args.random:
        features = _random_demo_flow()
    else:
        features = json.loads(Path(args.flow).read_text())

    result = FlowClassifier().predict(features)
    print(json.dumps(result, indent=2))


def cmd_enrich(args: argparse.Namespace) -> None:
    from mcp_server.ioc_intel import enrich_ioc

    result = enrich_ioc(args.indicator)
    print(json.dumps(result.__dict__, indent=2))


def cmd_forensics(args: argparse.Namespace) -> None:
    from mcp_server.pcap_forensics import analyze_pcap_summary

    print(json.dumps(analyze_pcap_summary(args.pcap), indent=2))


def cmd_attack(args: argparse.Namespace) -> None:
    from mcp_server.mitre_mapping import map_to_attack

    mapping = map_to_attack(args.label)
    if mapping is None:
        print(f"No ATT&CK mapping registered for label '{args.label}'.")
        return
    print(json.dumps(mapping.__dict__, indent=2))


def cmd_playbook(args: argparse.Namespace) -> None:
    from mcp_server.playbooks import propose_playbook

    pb = propose_playbook(args.label, args.confidence)
    print(json.dumps(pb.__dict__, indent=2))


def cmd_investigate(args: argparse.Namespace) -> None:
    if getattr(args, 'live', False):
        _cmd_investigate_live(args)
        return

    from agents.run_offline_demo import run_offline_incident

    features = json.loads(Path(args.flow).read_text()) if args.flow else _random_demo_flow()
    result = run_offline_incident(features, suspect_ip=args.ip, pcap_path=args.pcap)
    print(result["report_markdown"])

    sector = getattr(args, "sector", "Power & Energy")
    label = result.get("triage", {}).get("predicted_label", "")
    if label and label != "BENIGN":
        print("\n" + "=" * 65)
        print(f"  [!] NCIIPC CRITICAL INFORMATION INFRASTRUCTURE ADVISORY")
        print(f"      Designated Sector: {sector}")
        print(f"      Statutory Mandate: Section 70A, Information Technology Act, 2000")
        print(f"      Incident Contact:  helpdesk1@nciipc.gov.in | Portal: nciipc.gov.in")
        print(f"      Generate Advisory: python -m cli.argus_cli nciipc-report --sector \"{sector}\"")
        print("=" * 65)

    if args.out:
        Path(args.out).write_text(result["report_markdown"])
        print(f"\n(also written to {args.out})")


def _cmd_investigate_live(args: argparse.Namespace) -> None:
    """Run the real ADK + Gemini multi-agent pipeline from the CLI."""
    features = json.loads(Path(args.flow).read_text()) if args.flow else _random_demo_flow()

    parts = [
        "Investigate this network flow. Features:",
        ", ".join(f"{k}={v}" for k, v in features.items()) + ".",
    ]
    if args.ip:
        parts.append(f"Suspect IP: {args.ip}.")
    if args.pcap:
        parts.append(f"PCAP at: {args.pcap}.")
    description = " ".join(parts)

    from agents.run_live import run_incident
    print("\n🧠 Running live ADK + Gemini multi-agent pipeline...")
    print("   (this may take 30-60 seconds)\n")
    asyncio.run(run_incident(description))


def cmd_audit(args: argparse.Namespace) -> None:
    from security.guardrails import AuditLogger

    log = AuditLogger(PROJECT_ROOT / "data" / "audit_log.jsonl")
    if args.audit_action == "verify":
        ok = log.verify_chain()
        print("Audit log hash chain VALID" if ok else "Audit log hash chain BROKEN -- tampering detected")
        sys.exit(0 if ok else 1)
    elif args.audit_action == "tail":
        lines = log.path.read_text().splitlines() if log.path.exists() else []
        for line in lines[-args.n :]:
            entry = json.loads(line)
            print(f"[{entry['event_type']}] {entry['agent_role']}: {entry['detail']}")


def cmd_train_world_model(args: argparse.Namespace) -> None:
    from ml.world_model.train import train_world_model

    metrics = train_world_model(
        data_source=args.dataset,
        data_path=args.path,
        n_sequences=args.sequences,
        seq_len=getattr(args, 'seq_len', 10),
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    print(f"\nStage accuracy: {metrics['final_stage_accuracy']:.4f}")
    print(f"Infiltration AUC: {metrics['final_infiltration_auc']:.4f}")


def cmd_forecast(args: argparse.Namespace) -> None:
    from ml.world_model.predictor import forecast_infiltration

    if getattr(args, 'flow', None):
        features = json.loads(Path(args.flow).read_text())
    else:
        features = _random_demo_flow()

    result = forecast_infiltration(features, k_steps=args.steps)
    print(f"Risk Level: {result['risk_level']}")
    print(f"Max Infiltration Prob: {result['max_infiltration_prob']:.2%}")
    print(f"\nTimeline:")
    for i, (prob, stage) in enumerate(zip(
        result['probability_timeline'], result['predicted_stages']
    )):
        filled = int(prob * 20)
        bar = "#" * filled + "-" * (20 - filled)
        print(f"  T+{i+1}: [{bar}] {prob:.1%}  [{stage}]")
    print(f"\n{result['forecast_explanation']}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from ml.world_model.benchmark import run_benchmark, generate_benchmark_report

    results = run_benchmark(
        data_source=args.dataset,
        data_path=args.path,
        n_sequences=args.sequences,
    )
    report = generate_benchmark_report(results)
    print("\n" + report)


def cmd_datasets(args: argparse.Namespace) -> None:
    from ml.world_model.dataset_loader import list_datasets

    registry = list_datasets()
    print("\n\033[1mARGUS Supported Public Cybersecurity Datasets\033[0m")
    print("=" * 65)
    for key, info in registry.items():
        print(f"\n  \033[96m{key}\033[0m: {info['name']}")
        print(f"    {info['description'][:80]}")
        if info.get('url'):
            print(f"    URL:  {info['url']}")
        if info.get('size'):
            print(f"    Size: {info['size']}")
    print()


def cmd_kb_enrich(args: argparse.Namespace) -> None:
    from mcp_server.knowledge_base import enrich_detection_with_kb

    result = enrich_detection_with_kb(args.label, args.technique)

    print(f"\n\033[1mKnowledge Base Enrichment: {args.label}\033[0m")
    print("=" * 55)

    techs = result.get("attack_techniques", [])
    if techs:
        print(f"\n\033[93mMITRE ATT&CK Techniques ({len(techs)}):\033[0m")
        for t in techs:
            print(f"  {t['technique_id']}: {t['name']} ({t['tactic']})")
            if t.get('mitigations'):
                for m in t['mitigations'][:2]:
                    print(f"    -> Mitigation: {m}")
            if t.get('url'):
                print(f"    -> {t['url']}")

    patterns = result.get("capec_patterns", [])
    if patterns:
        print(f"\n\033[93mCAPEC Attack Patterns ({len(patterns)}):\033[0m")
        for p in patterns:
            print(f"  {p['capec_id']}: {p['name']}")
            if p.get('severity'):
                print(f"    Severity: {p['severity']}")
            if p.get('related_cwe'):
                print(f"    Related CWE: {', '.join(p['related_cwe'][:3])}")

    cves = result.get("related_cves", [])
    if cves:
        print(f"\n\033[93mRelated CVEs ({len(cves)}):\033[0m")
        for c in cves:
            score = f"CVSS {c['cvss_score']}" if c.get('cvss_score') else "N/A"
            print(f"  {c['cve_id']}: {score} ({c.get('severity', 'N/A')})")
            if c.get('description'):
                print(f"    {c['description'][:80]}...")
    print()


def cmd_kb_status(args: argparse.Namespace) -> None:
    from mcp_server.knowledge_base import get_kb_status

    status = get_kb_status()
    print("\n\033[1mARGUS Knowledge Base Status\033[0m")
    print("=" * 45)
    for kb_name, info in status.items():
        cached = info.get("cache_exists", False)
        icon = "[OK]" if cached else "[--]"
        print(f"\n  {icon} {kb_name}:")
        for k, v in info.items():
            print(f"    {k}: {v}")
    print()


def cmd_lookup_cve(args: argparse.Namespace) -> None:
    from mcp_server.knowledge_base import lookup_cve

    result = lookup_cve(args.cve_id)
    if not result:
        print(f"CVE {args.cve_id} not found.")
        return

    print(f"\n\033[1m{result['cve_id']}\033[0m")
    print(f"  Severity: {result.get('severity', 'N/A')}")
    print(f"  CVSS:     {result.get('cvss_score', 'N/A')}")
    print(f"  Published: {result.get('published', 'N/A')}")
    if result.get('description'):
        print(f"  {result['description'][:200]}")
    if result.get('affected_products'):
        print(f"  Products: {', '.join(result['affected_products'][:5])}")
    if result.get('references'):
        print(f"  References:")
        for ref in result['references'][:3]:
            print(f"    {ref}")
    print()


def cmd_nciipc_report(args: argparse.Namespace) -> None:
    from ml.model import FlowClassifier
    from mcp_server.nciipc_report import generate_nciipc_report
    from ml.world_model.predictor import forecast_infiltration

    if getattr(args, "random", False) or not getattr(args, "flow", None):
        features = _random_demo_flow()
    else:
        features = json.loads(Path(args.flow).read_text())

    clf = FlowClassifier()
    det = clf.predict(features)
    forecast = forecast_infiltration(features, k_steps=args.steps)

    report = generate_nciipc_report(
        predicted_label=det["predicted_label"],
        confidence=det["confidence"],
        cii_sector=args.sector,
        forecast=forecast,
        suspect_ips=[args.ip] if args.ip else ["185.220.101.45"],
        affected_ips=["10.0.1.5"],
    )

    print(report.report_markdown)
    if args.email:
        print("\n" + "=" * 60)
        print("EMAIL TRANSMISSION DRAFT (helpdesk1@nciipc.gov.in)")
        print("=" * 60)
        print(report.nciipc_email_draft)
    if args.out:
        Path(args.out).write_text(report.report_markdown, encoding="utf-8")
        print(f"\n[OK] NCIIPC Incident Report saved to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argus", description="ARGUS SOC Agent Skills CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the Random Forest flow classifier")
    p_train.add_argument("--n-per-class", type=int, default=1500)
    p_train.add_argument("--fast", action="store_true", help="Skip grid search hyperparameter tuning")
    p_train.set_defaults(func=cmd_train)

    p_detect = sub.add_parser("detect", help="Classify a network flow")
    g = p_detect.add_mutually_exclusive_group(required=True)
    g.add_argument("--flow", help="Path to a JSON file of flow features")
    g.add_argument("--random", action="store_true", help="Classify a random synthetic demo flow")
    p_detect.set_defaults(func=cmd_detect)

    p_enrich = sub.add_parser("enrich", help="Threat-intel lookup for an IP/domain")
    p_enrich.add_argument("indicator")
    p_enrich.set_defaults(func=cmd_enrich)

    p_forensics = sub.add_parser("forensics", help="Summarize a PCAP file")
    p_forensics.add_argument("pcap")
    p_forensics.set_defaults(func=cmd_forensics)

    p_attack = sub.add_parser("attack", help="MITRE ATT&CK lookup for a detection label")
    p_attack.add_argument("label")
    p_attack.set_defaults(func=cmd_attack)

    p_playbook = sub.add_parser("playbook", help="Propose a remediation playbook")
    p_playbook.add_argument("label")
    p_playbook.add_argument("--confidence", type=float, default=0.9)
    p_playbook.set_defaults(func=cmd_playbook)

    p_inv = sub.add_parser("investigate", help="Run the full pipeline on one incident")
    inv_g = p_inv.add_mutually_exclusive_group()
    inv_g.add_argument("--flow", help="Path to a JSON flow-features file")
    inv_g.add_argument("--random", action="store_true", help="Use a random synthetic demo flow (default if --flow is omitted)")
    p_inv.add_argument("--pcap", help="Path to a PCAP file to summarize")
    p_inv.add_argument("--ip", help="Suspect IP/domain to enrich")
    p_inv.add_argument("--sector", default="Power & Energy", help="Designated CII sector for NCIIPC advisory")
    p_inv.add_argument("--out", help="Write the Markdown report to this path")
    p_inv.add_argument("--live", action="store_true", help="Use real ADK + Gemini agents instead of the offline pipeline")
    p_inv.set_defaults(func=cmd_investigate)

    p_audit = sub.add_parser("audit", help="Inspect/verify the agent audit log")
    p_audit.add_argument("audit_action", choices=["verify", "tail"])
    p_audit.add_argument("--n", type=int, default=10, help="Lines to show for 'tail'")
    p_audit.set_defaults(func=cmd_audit)

    # --- World Model commands ---
    _all_datasets = ["synthetic", "cicids2017", "cicids2018", "unsw_nb15", "ctu13", "ciciot2023", "lanl", "darpa"]

    p_train_wm = sub.add_parser("train-world-model", help="Train the World Model Transformer")
    p_train_wm.add_argument("--dataset", default="synthetic", choices=_all_datasets)
    p_train_wm.add_argument("--path", default=None, help="Path to dataset directory/file")
    p_train_wm.add_argument("--epochs", type=int, default=50)
    p_train_wm.add_argument("--lr", type=float, default=1e-3)
    p_train_wm.add_argument("--sequences", type=int, default=500)
    p_train_wm.add_argument("--seq-len", type=int, default=10)
    p_train_wm.add_argument("--device", default="auto")
    p_train_wm.set_defaults(func=cmd_train_world_model)

    p_forecast = sub.add_parser("forecast", help="Run K-step infiltration forecast")
    fc_g = p_forecast.add_mutually_exclusive_group()
    fc_g.add_argument("--flow", help="Path to a JSON flow-features file")
    fc_g.add_argument("--random", action="store_true", help="Use a random synthetic demo flow")
    p_forecast.add_argument("--steps", type=int, default=5, help="Number of future steps to simulate")
    p_forecast.set_defaults(func=cmd_forecast)

    p_benchmark = sub.add_parser("benchmark", help="Benchmark World Model vs Logistic Regression")
    p_benchmark.add_argument("--dataset", default="synthetic", choices=_all_datasets)
    p_benchmark.add_argument("--path", default=None)
    p_benchmark.add_argument("--sequences", type=int, default=500)
    p_benchmark.set_defaults(func=cmd_benchmark)

    # --- Knowledge Base & Dataset commands ---
    p_datasets = sub.add_parser("datasets", help="List all supported public cybersecurity datasets")
    p_datasets.set_defaults(func=cmd_datasets)

    p_kb_enrich = sub.add_parser("kb-enrich", help="Enrich a label with ATT&CK/CAPEC/CVE context")
    p_kb_enrich.add_argument("label", help="Detection label (e.g. DDoS, BruteForce)")
    p_kb_enrich.add_argument("--technique", default=None, help="Specific ATT&CK technique ID")
    p_kb_enrich.set_defaults(func=cmd_kb_enrich)

    p_kb_status = sub.add_parser("kb-status", help="Show knowledge base cache status")
    p_kb_status.set_defaults(func=cmd_kb_status)

    p_lookup_cve = sub.add_parser("lookup-cve", help="Look up a specific CVE from NVD")
    p_lookup_cve.add_argument("cve_id", help="CVE ID (e.g. CVE-2021-44228)")
    p_lookup_cve.set_defaults(func=cmd_lookup_cve)

    p_nciipc = sub.add_parser("nciipc-report", help="Generate official NCIIPC Critical Information Infrastructure report")
    nciipc_g = p_nciipc.add_mutually_exclusive_group()
    nciipc_g.add_argument("--flow", help="Path to a JSON flow-features file")
    nciipc_g.add_argument("--random", action="store_true", help="Use a random synthetic demo flow (default)")
    p_nciipc.add_argument("--sector", default="Power & Energy", help="CII Sector (Power, BFSI, Telecom, Transport, Government, Defense)")
    p_nciipc.add_argument("--steps", type=int, default=5, help="Number of future forecast steps")
    p_nciipc.add_argument("--ip", help="Suspect IP to include in report")
    p_nciipc.add_argument("--email", action="store_true", help="Print draft email formatted for helpdesk1@nciipc.gov.in")
    p_nciipc.add_argument("--out", help="Save report markdown to file")
    p_nciipc.set_defaults(func=cmd_nciipc_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

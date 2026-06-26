#!/usr/bin/env python3
"""
argus_cli.py
-------------
ARGUS Agent Skills CLI -- each subcommand is a discrete, independently
invokable "agent skill" that wraps one stage of the SOC pipeline, so an
analyst (or another automation, e.g. a SOAR playbook or cron job) can call
exactly the capability they need without spinning up the full multi-agent
session. This is the capstone rubric's "Agent skills (e.g. Agents CLI)"
artifact.

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
    p_inv.add_argument("--out", help="Write the Markdown report to this path")
    p_inv.add_argument("--live", action="store_true", help="Use real ADK + Gemini agents instead of the offline pipeline")
    p_inv.set_defaults(func=cmd_investigate)

    p_audit = sub.add_parser("audit", help="Inspect/verify the agent audit log")
    p_audit.add_argument("audit_action", choices=["verify", "tail"])
    p_audit.add_argument("--n", type=int, default=10, help="Lines to show for 'tail'")
    p_audit.set_defaults(func=cmd_audit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_sample_datasets.py
----------------------------
Generates bundled representative sample fixtures for all 7 public cybersecurity
datasets in argus/data/sample_datasets/. This allows ARGUS to test, demonstrate,
and benchmark all 8 datasets immediately without gigabyte downloads.
"""

from pathlib import Path
import numpy as np
import pandas as pd

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_datasets"


def create_cicids2018_sample(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = ["Benign", "DoS attacks-Hulk", "FTP-BruteForce", "Bot", "Infilteration", "SQL Injection"] * 5
    for i, label in enumerate(labels):
        is_attack = label != "Benign"
        rows.append({
            "Flow Duration": 5000 + i * 200,
            "Total Fwd Packet": 15 + (100 if is_attack else 5),
            "Total Bwd packets": 12 + (80 if is_attack else 4),
            "Total Length of Fwd Packet": 1500 + (25000 if is_attack else 200),
            "Total Length of Bwd Packet": 1200 + (18000 if is_attack else 150),
            "Fwd Packet Length Mean": 350.0 + (300.0 if is_attack else 20.0),
            "Fwd Packet Length Std": 50.0,
            "Bwd Packet Length Mean": 320.0 + (250.0 if is_attack else 15.0),
            "Bwd Packet Length Std": 45.0,
            "Flow Bytes/s": 4000.0 + (15000.0 if is_attack else 100.0),
            "Flow Packets/s": 25.0 + (120.0 if is_attack else 5.0),
            "Flow IAT Mean": 45.0,
            "Flow IAT Std": 15.0,
            "Fwd IAT Mean": 40.0,
            "Bwd IAT Mean": 42.0,
            "SYN Flag Count": 1 if not is_attack else 12,
            "ACK Flag Count": 15 if not is_attack else 80,
            "RST Flag Count": 0 if not is_attack else 8,
            "PSH Flag Count": 4,
            "FIN Flag Count": 1,
            "Fwd Header Length": 400,
            "Bwd Header Length": 350,
            "Down/Up Ratio": 0.85,
            "Average Packet Size": 450.0,
            "Fwd IAT Std": 12.0,
            "Bwd IAT Std": 14.0,
            "Flow IAT Max": 120.0,
            "URG Flag Count": 0,
            "Label": label,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sample_cicids2018.csv", index=False)
    print(f"Created {out_dir / 'sample_cicids2018.csv'} ({len(df)} rows)")


def create_cicids2017_sample(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = ["BENIGN", "DDoS", "PortScan", "Bot", "FTP-Patator", "Infiltration"] * 5
    for i, label in enumerate(labels):
        is_attack = label != "BENIGN"
        rows.append({
            " Flow Duration": 4800 + i * 150,
            " Total Fwd Packets": 14 + (90 if is_attack else 4),
            " Total Backward Packets": 11 + (70 if is_attack else 3),
            "Total Length of Fwd Packets": 1400 + (22000 if is_attack else 180),
            "Total Length of Bwd Packets": 1100 + (16000 if is_attack else 140),
            "Fwd Packet Length Mean": 340.0 + (280.0 if is_attack else 18.0),
            "Fwd Packet Length Std": 48.0,
            "Bwd Packet Length Mean": 310.0 + (230.0 if is_attack else 14.0),
            "Bwd Packet Length Std": 42.0,
            "Flow Bytes/s": 3800.0 + (14000.0 if is_attack else 90.0),
            "Flow Packets/s": 22.0 + (110.0 if is_attack else 4.0),
            "Flow IAT Mean": 44.0,
            "Flow IAT Std": 14.0,
            "Fwd IAT Mean": 39.0,
            "Bwd IAT Mean": 41.0,
            "SYN Flag Count": 1 if not is_attack else 10,
            "ACK Flag Count": 14 if not is_attack else 75,
            "RST Flag Count": 0 if not is_attack else 6,
            "PSH Flag Count": 3,
            "FIN Flag Count": 1,
            "URG Flag Count": 0,
            "Down/Up Ratio": 0.82,
            "Average Packet Size": 440.0,
            "Fwd Header Length": 380,
            "Bwd Header Length": 340,
            "Fwd IAT Std": 11.0,
            "Bwd IAT Std": 13.0,
            "Flow IAT Max": 115.0,
            " Label": label,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sample_cicids2017.csv", index=False)
    print(f"Created {out_dir / 'sample_cicids2017.csv'} ({len(df)} rows)")


def create_unsw_nb15_sample(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cats = ["Normal", "Exploits", "Reconnaissance", "DoS", "Backdoor", "Shellcode"] * 5
    for i, cat in enumerate(cats):
        is_attack = cat != "Normal"
        rows.append({
            "dur": 1.2 + i * 0.1,
            "spkts": 20 + (80 if is_attack else 5),
            "dpkts": 18 + (70 if is_attack else 4),
            "sbytes": 1800 + (15000 if is_attack else 300),
            "dbytes": 1600 + (12000 if is_attack else 250),
            "sttl": 64,
            "dttl": 62,
            "swin": 255,
            "dwin": 255,
            "smean": 90.0 + (120.0 if is_attack else 10.0),
            "dmean": 88.0 + (110.0 if is_attack else 8.0),
            "sjit": 12.0,
            "djit": 10.0,
            "sinpkt": 15.0,
            "dinpkt": 14.0,
            "tcprtt": 0.045,
            "ct_dst_sport_ltm": 2 + (15 if is_attack else 0),
            "ct_flw_http_mthd": 1 if is_attack else 0,
            "attack_cat": cat,
            "label": 1 if is_attack else 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sample_unsw_nb15.csv", index=False)
    print(f"Created {out_dir / 'sample_unsw_nb15.csv'} ({len(df)} rows)")


def create_ctu13_sample(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label"
    ]
    for i in range(30):
        is_bot = (i % 3 == 1)
        dur = 2.5 + i * 0.2
        pkts = 150 if is_bot else 15
        bytes_ = 45000 if is_bot else 1200
        src_bytes = 25000 if is_bot else 600
        lbl = "flow=From-Botnet-V42-TCP-Attempt" if is_bot else "flow=Normal-TCP-Established"
        lines.append(f"2011/08/10 09:12:{i:02d}.100,{dur:.3f},tcp,147.32.84.165,102{i},->,147.32.96.1,80,CON,0,0,{pkts},{bytes_},{src_bytes},{lbl}")
    (out_dir / "sample_ctu13.binetflow").write_text("\n".join(lines))
    print(f"Created {out_dir / 'sample_ctu13.binetflow'} (30 rows)")


def create_ciciot2023_sample(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    labels = ["BenignTraffic", "DDoS-UDP_Flood", "Recon-PortScan", "Mirai-greip_flood", "BruteForce"] * 6
    for i, label in enumerate(labels):
        is_attack = label != "BenignTraffic"
        rows.append({
            "flow_duration": 4.5 + i * 0.2,
            "Header_Length": 540,
            "Rate": 45.0 + (300.0 if is_attack else 10.0),
            "Srate": 3500.0 + (25000.0 if is_attack else 150.0),
            "fin_flag_number": 1,
            "syn_flag_number": 1 if not is_attack else 15,
            "rst_flag_number": 0 if not is_attack else 5,
            "psh_flag_number": 4,
            "ack_flag_number": 18 if not is_attack else 90,
            "urg_flag_number": 0,
            "Tot sum": 2500.0 + (35000.0 if is_attack else 300.0),
            "Min": 40.0,
            "Max": 1200.0,
            "AVG": 450.0,
            "Std": 85.0,
            "Tot size": 2200.0 + (30000.0 if is_attack else 250.0),
            "IAT": 48.0,
            "Number": 22 + (120 if is_attack else 5),
            "Magnitue": 4000.0,
            "Radius": 64.0,
            "Covariance": 12.0,
            "Variance": 45.0,
            "Weight": 0.88,
            "label": label,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sample_ciciot2023.csv", index=False)
    print(f"Created {out_dir / 'sample_ciciot2023.csv'} ({len(df)} rows)")


def create_lanl_sample(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(120):
        t = 100 + (i // 4) * 60 + (i % 4) * 2
        user = f"U{100 + (i % 8)}@DOM1"
        src_comp = f"C{200 + (i % 5)}"
        dst_comp = f"C{300 + (i % 12)}"
        success = "Fail" if (20 <= i <= 50 and i % 2 == 0) else "Success"
        lines.append(f"{t},{user},{user},{src_comp},{dst_comp},Negotiate,Network,LogOn,{success}")
    (out_dir / "sample_auth.txt").write_text("\n".join(lines))
    print(f"Created {out_dir / 'sample_auth.txt'} (120 lines)")


def create_darpa_sample(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    kdd_columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label",
    ]
    labels = ["normal", "neptune", "ipsweep", "guess_passwd", "buffer_overflow"] * 6
    rows = []
    for i, label in enumerate(labels):
        is_attack = label != "normal"
        rows.append({
            "duration": 0 if not is_attack else 5,
            "protocol_type": "tcp",
            "service": "http" if not is_attack else "private",
            "flag": "SF" if not is_attack else "S0",
            "src_bytes": 240 if not is_attack else 4500,
            "dst_bytes": 1800 if not is_attack else 0,
            "land": 0,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0 if not is_attack else 3,
            "logged_in": 1 if not is_attack else 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_shells": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0,
            "count": 12 if not is_attack else 120,
            "srv_count": 10 if not is_attack else 115,
            "serror_rate": 0.0 if not is_attack else 0.95,
            "srv_serror_rate": 0.0 if not is_attack else 0.95,
            "rerror_rate": 0.0,
            "srv_rerror_rate": 0.0,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "srv_diff_host_rate": 0.0,
            "dst_host_count": 25,
            "dst_host_srv_count": 25,
            "dst_host_same_srv_rate": 1.0,
            "dst_host_diff_srv_rate": 0.0,
            "dst_host_same_src_port_rate": 0.05,
            "dst_host_srv_diff_host_rate": 0.0,
            "dst_host_serror_rate": 0.0 if not is_attack else 0.9,
            "dst_host_srv_serror_rate": 0.0 if not is_attack else 0.9,
            "dst_host_rerror_rate": 0.0,
            "dst_host_srv_rerror_rate": 0.0,
            "label": label,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sample_nsl_kdd.csv", index=False)
    print(f"Created {out_dir / 'sample_nsl_kdd.csv'} ({len(df)} rows)")


def main():
    print("Generating sample fixtures for all 7 public datasets...")
    create_cicids2018_sample(SAMPLE_DIR / "cicids2018")
    create_cicids2017_sample(SAMPLE_DIR / "cicids2017")
    create_unsw_nb15_sample(SAMPLE_DIR / "unsw_nb15")
    create_ctu13_sample(SAMPLE_DIR / "ctu13")
    create_ciciot2023_sample(SAMPLE_DIR / "ciciot2023")
    create_lanl_sample(SAMPLE_DIR / "lanl")
    create_darpa_sample(SAMPLE_DIR / "darpa")
    print(f"All sample fixtures generated successfully in {SAMPLE_DIR}")


if __name__ == "__main__":
    main()

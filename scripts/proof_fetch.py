"""Local side of the proof report: fetch S3DF evidence PNGs listed in <remote dir>/evidence_auto.json
into results/proof_report/ and merge with results/proof_report/evidence_manual.json -> evidence.json.
usage (repo root): python3 scripts/proof_fetch.py <bridge-session> <remote dir>
"""
import json
import os
import subprocess
import sys

S, R = sys.argv[1], sys.argv[2]
auto = json.loads(subprocess.run(["bridge", "--session", S, "read", os.path.join(R, "evidence_auto.json"), "--raw"],
                                 capture_output=True, text=True, check=True).stdout)
os.makedirs("results/proof_report", exist_ok=True)
for f in auto["figures"]:
    subprocess.run(["ops/fetch_bin.sh", S, f["remote"], f["file"]], check=True, capture_output=True)
    ok = open(f["file"], "rb").read(8) == b"\x89PNG\r\n\x1a\n" and os.path.getsize(f["file"]) < 3 * 2 ** 20
    print(("ok  " if ok else "BAD ") + f["file"], os.path.getsize(f["file"]))
man = json.load(open("results/proof_report/evidence_manual.json")) if os.path.isfile("results/proof_report/evidence_manual.json") else {}
ev = dict(man)
ev["figures"] = [{k: v for k, v in f.items() if k != "remote"} for f in auto["figures"]] + man.get("figures", [])
json.dump(ev, open("results/proof_report/evidence.json", "w"), indent=1)
print("evidence.json: %d figures" % len(ev["figures"]))

"""Assemble results/proof_report/index.html (one self-contained page, images inlined) from
  - results/<statement>/result.json of every model statement (+ detection, leaderboard, split)
  - results/proof_report/evidence.json: S3DF evidence figures (made by scripts/proof_figs.py and
    seg/viz.py on S3DF, fetched with ops/fetch_bin.sh), independent re-evaluations
    (scripts/independent_reeval.py on S3DF), findings, not-covered items, audit notes.
Every image in the page is an S3DF artefact. Stdlib only; run in the repo root:
    python3 scripts/make_proof_report.py [--commit SHA]
"""
import argparse
import base64
import html
import json
import os
import time

MODELS = ["peaknet_baseline", "hf_segformer", "hf_upernet", "lucid_segformer", "lucid_unet", "best_model"]
OUT_DIR = "results/proof_report"
BUDGET = 14 * 2 ** 20  # keep the page under the 15 MB check limit

CSS = """
:root{--bg:#F4F6F8;--surface:#FFF;--fg:#141A22;--muted:#566170;--line:#D8DEE6;--accent:#1D5BB8;--pass:#1B7442;--warn:#9A5600;--fail:#B42318;--code:#EBEFF4;
--sans:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#0D1116;--surface:#141A21;--fg:#E3E8EE;--muted:#98A3B1;--line:#28313C;--accent:#7DB0FF;--pass:#5DC98C;--warn:#E2A54C;--fail:#FF8A80;--code:#1A2129;color-scheme:dark}}
body{background:var(--bg);color:var(--fg);font:15px/1.6 var(--sans);margin:0}
.wrap{max-width:1180px;margin:0 auto;padding:28px 16px 56px;display:grid;gap:40px}
section{display:grid;gap:14px;min-width:0}
h1,h2,h3{font-weight:600;line-height:1.2;margin:0}h1{font-size:2rem}h2{font-size:1.3rem}h3{font-size:1.02rem}
p{margin:0;max-width:80ch}.eyebrow{font:500 .78rem/1.4 var(--mono);color:var(--muted);letter-spacing:.04em;text-transform:uppercase}
.lede{font-size:1.08rem}.muted{color:var(--muted)}code,.mono{font-family:var(--mono);font-size:.86em}
code{background:var(--code);padding:.1em .35em;border-radius:4px;word-break:break-all}a{color:var(--accent)}
.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.verdict{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px;display:grid;gap:6px;align-content:start}
.verdict .k{font-weight:600;display:flex;justify-content:space-between;gap:8px;align-items:baseline}
.verdict .v{font-family:var(--mono);font-size:.84rem;color:var(--muted)}
.chip{font:600 .72rem/1 var(--mono);letter-spacing:.06em;padding:4px 7px;border-radius:4px;border:1px solid currentColor;white-space:nowrap}
.chip.pass{color:var(--pass)}.chip.warn{color:var(--warn)}.chip.fail{color:var(--fail)}.chip.notrun{color:var(--muted)}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:0}th{font:500 .74rem/1.3 var(--mono);text-transform:uppercase;letter-spacing:.05em;color:var(--muted);white-space:nowrap}
td.num{font-family:var(--mono);white-space:nowrap}td small{color:var(--muted);display:block;margin-top:4px}
.plots{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:18px}
figure{margin:0;display:grid;gap:8px;align-content:start;min-width:0}
figure img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line);background:#fff}
figcaption{font-size:.84rem;color:var(--muted)}figcaption b{color:var(--fg);font-family:var(--mono);font-size:.92em}
.model{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px;display:grid;gap:12px}
ul.findings{margin:0;padding-left:1.3em;display:grid;gap:10px;max-width:85ch}
pre{margin:0;background:var(--code);border-radius:8px;padding:14px 16px;overflow-x:auto;font:.82rem/1.55 var(--mono)}
details{border-top:1px solid var(--line);padding-top:14px}details summary{cursor:pointer;font-weight:600}details>div{display:grid;gap:10px;margin-top:12px}
"""


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def esc(x):
    return html.escape(str(x))


def f4(x):
    return "%.4f" % x if isinstance(x, (int, float)) else "-"


class Images:
    def __init__(self):
        self.used = 0

    def tag(self, path, alt):
        if not os.path.isfile(path):
            return '<p class="muted">missing image %s</p>' % esc(path)
        raw = open(path, "rb").read()
        if self.used + len(raw) * 4 // 3 < BUDGET:
            self.used += len(raw) * 4 // 3
            src = "data:image/png;base64," + base64.b64encode(raw).decode()
        else:  # over budget: link the PNG that sits next to the page in the repo
            src = os.path.relpath(path, OUT_DIR)
        return '<img loading="lazy" alt="%s" src="%s">' % (esc(alt), src)


def fig(imgs, path, caption, alt=None):
    return "<figure>%s<figcaption>%s</figcaption></figure>" % (imgs.tag(path, alt or caption), caption)


def chip(v):
    return '<span class="chip %s">%s</span>' % ({"PASS": "pass", "WARN": "warn", "FAIL": "fail"}.get(v, "notrun"), v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="")
    a = ap.parse_args()
    ev = load(os.path.join(OUT_DIR, "evidence.json")) or {}
    split = load("results/split/result.json") or {}
    n_test = split.get("n_images", {}).get("test")
    res = {m: load("results/%s/result.json" % m) for m in MODELS}
    pending = set()
    for m in MODELS:  # a statement that lands after this page (ORDER 20): shown from its final S3DF eval
        if not res[m] and os.path.isfile(os.path.join(OUT_DIR, "pending_%s.json" % m)):
            res[m] = load(os.path.join(OUT_DIR, "pending_%s.json" % m))
            pending.add(m)
    det = load("results/detection/result.json")
    imgs = Images()
    now = time.strftime("%Y-%m-%d %H:%M %Z")
    landed = [m for m in MODELS if res[m]]
    best = max(landed, key=lambda m: res[m]["eval"]["peak"]["f1"]) if landed else None

    H = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>PeakNet segmentation sprint: proof report</title><style>%s</style></head><body><div class=\"wrap\">" % CSS]
    # header
    H.append('<section><div class="eyebrow">S3DF (SLAC) &middot; %s &middot; Slurm allocation(s) %s &middot; repo main %s &middot; page built %s</div>'
             % (esc(ev.get("cluster", "ada partition, NVIDIA L40S")), esc(", ".join(ev.get("jobs", [])) or "see squeue capture"),
                esc(a.commit or "?"), esc(now)))
    H.append("<h1>Bragg-peak segmentation on peaknet10k: proof report</h1>")
    lede = ("Six segmentation statements (%s) trained on the shared run-level split and evaluated on the <b>whole test split (%s images)</b> "
            "with the shared metric code (test peak F1 at 3 px tolerance; pixel IoU second). "
            % (", ".join("<code>%s</code>" % m for m in MODELS), esc(n_test)))
    if best:
        lede += "Best so far: <code>%s</code> at test peak F1 <b>%.4f</b>. " % (best, res[best]["eval"]["peak"]["f1"])
    lede += "Every figure on this page was rendered on S3DF from real checkpoints, predictions and logs; nothing was captured on any other machine."
    H.append('<p class="lede">%s</p></section>' % lede)

    # verdict cards
    H.append('<section><h2>Verdicts per statement</h2><div class="verdicts">')
    for m in MODELS:
        d = res[m]
        if not d:
            H.append('<div class="verdict"><div class="k"><code>%s</code>%s</div><div class="v">no result on main when this page was built</div></div>' % (m, chip("NOT RUN")))
            continue
        e = d["eval"]
        ok = e.get("n_images") == n_test and e.get("peak_tol_px") == 3
        H.append('<div class="verdict"><div class="k"><code>%s</code>%s</div><div>%s</div><div class="v">peak F1 %s &middot; P %s &middot; R %s<br>pixel IoU %s &middot; %s test images</div></div>'
                 % (m, chip("PASS" if ok else "WARN"), esc(d.get("model", "")) + (" <small>(lands on main last, after this page; numbers from its final whole-test eval on S3DF)</small>" if m in pending else ""), f4(e["peak"]["f1"]), f4(e["peak"].get("precision")),
                    f4(e["peak"].get("recall")), f4(e["pixel"]["iou"]), esc(e.get("n_images"))))
    if det:
        e = det["eval"]
        H.append('<div class="verdict"><div class="k"><code>detection</code>%s</div><div>%s</div><div class="v">mAP50 %s &middot; P %s &middot; R %s<br>%s test images</div></div>'
                 % (chip("PASS" if e.get("n_images") == n_test else "WARN"), esc(det.get("model", "")), f4(e.get("map50")),
                    f4(e.get("precision")), f4(e.get("recall")), esc(e.get("n_images"))))
    H.append("</div></section>")

    # leaderboard
    rows = sorted([m for m in landed], key=lambda m: -res[m]["eval"]["peak"]["f1"])
    H.append('<section><h2>Leaderboard (whole test split)</h2><div class="table-wrap"><table><tr><th>#</th><th>statement</th><th>model</th>'
             "<th>peak F1</th><th>peak P</th><th>peak R</th><th>pixel IoU</th><th>test images</th><th>train</th></tr>")
    for i, m in enumerate(rows, 1):
        d, e = res[m], res[m]["eval"]
        tr = d.get("train", {})
        H.append("<tr><td class=num>%d</td><td><code>%s</code></td><td>%s</td><td class=num>%s</td><td class=num>%s</td><td class=num>%s</td><td class=num>%s</td><td class=num>%s</td><td><small>job %s &middot; %s GPU &middot; %s min</small></td></tr>"
                 % (i, m, esc(d.get("model", "")), f4(e["peak"]["f1"]), f4(e["peak"].get("precision")), f4(e["peak"].get("recall")),
                    f4(e["pixel"]["iou"]), esc(e.get("n_images")), esc(tr.get("slurm_jobid")), esc(tr.get("gpus")), esc(tr.get("wallclock_min"))))
    for m in MODELS:
        if not res[m]:
            H.append("<tr><td></td><td><code>%s</code></td><td class=muted colspan=7>not on main yet</td></tr>" % m)
    H.append("</table></div>")
    for f in ev.get("figures", []):
        if f.get("section") == "leaderboard":
            H.append(fig(imgs, f["file"], f["caption"]))
    H.append("</section>")

    # checks table
    H.append('<section><h2>Checks</h2><div class="table-wrap"><table><tr><th>check</th><th>expected</th><th>measured</th><th>verdict</th></tr>')
    for m in landed:
        e = res[m]["eval"]
        H.append("<tr><td><code>%s</code> evaluated on the whole test split</td><td class=num>n_images = %s, tol 3 px</td><td class=num>n_images = %s, tol %s</td><td>%s</td></tr>"
                 % (m, esc(n_test), esc(e.get("n_images")), esc(e.get("peak_tol_px")), chip("PASS" if e.get("n_images") == n_test and e.get("peak_tol_px") == 3 else "FAIL")))
    for r in ev.get("reeval", []):
        rep, got = r.get("reported_f1"), r.get("peak", {}).get("f1")
        ok = isinstance(rep, (int, float)) and isinstance(got, (int, float)) and abs(rep - got) <= r.get("tolerance", 2e-3) and r.get("n_images_scored") == n_test
        H.append("<tr><td>independent re-evaluation of <code>%s</code> (separate scorer, labels read from Zarr)</td><td class=num>peak F1 = %s &plusmn; %s on %s images</td><td class=num>peak F1 = %s on %s images</td><td>%s</td></tr>"
                 % (esc(r.get("statement")), f4(rep), r.get("tolerance", 2e-3), esc(n_test), f4(got), esc(r.get("n_images_scored")), chip("PASS" if ok else "WARN")))
    for c in ev.get("checks", []):
        H.append("<tr><td>%s</td><td class=num>%s</td><td class=num>%s</td><td>%s</td></tr>" % (c["check"], esc(c["expected"]), esc(c["measured"]), chip(c["verdict"])))
    if best:
        others = [m for m in landed if m != "best_model"]
        if res.get("best_model"):
            bf = res["best_model"]["eval"]["peak"]["f1"]
            mx = max([res[m]["eval"]["peak"]["f1"] for m in others] or [0])
            H.append("<tr><td><code>best_model</code> beats every other model statement</td><td class=num>&ge; %s</td><td class=num>%s</td><td>%s</td></tr>"
                     % (f4(mx), f4(bf), chip("PASS" if bf >= mx else "FAIL")))
    H.append("</table></div></section>")

    # per-model sections
    H.append("<section><h2>Models: annotated test predictions and provenance</h2>")
    H.append('<p class="muted">Overlays are made by <code>seg/viz.py</code> on S3DF from real test predictions: full detector image with every peak circled '
             "(green = matched within 3 px, red = missed, yellow = false), zoom crops with the ground-truth mask (cyan) and the predicted mask (magenta), "
             "image id and metrics written on the figure.</p>")
    for m in MODELS:
        d = res[m]
        H.append('<div class="model"><h3><code>%s</code>%s</h3>' % (m, (" &mdash; " + esc(d.get("model", ""))) if d else ""))
        if m in pending:
            H.append('<p class="muted">This statement lands on main after this page (the run closes when it lands). Shown from its final whole-test evaluation on S3DF; the leaderboard on main is regenerated in the same commit that lands it.</p>')
        if not d:
            H.append('<p class="muted">No result on main when this page was built.</p></div>')
            continue
        tr, e = d.get("train", {}), d["eval"]
        up = ", ".join('<a href="%s">%s</a> <small>(%s)</small>' % (esc(u.get("url")), esc(u.get("url")), esc(u.get("ref"))) for u in d.get("upstream", []))
        H.append('<div class="table-wrap"><table>'
                 "<tr><th>upstream</th><td>%s</td></tr>"
                 "<tr><th>train</th><td>Slurm step <code>%s</code>, %s GPU, %s min, %s training images, config <code>%s</code><br>log <code>%s</code></td></tr>"
                 "<tr><th>checkpoint</th><td><code>%s</code></td></tr>"
                 "<tr><th>eval</th><td><code>%s</code><br>log <code>%s</code></td></tr>"
                 "<tr><th>test metrics</th><td class=num>peak F1 %s (P %s, R %s) &middot; pixel IoU %s, F1 %s &middot; %s images</td></tr>"
                 "%s</table></div>"
                 % (up, esc(tr.get("slurm_jobid")), esc(tr.get("gpus")), esc(tr.get("wallclock_min")), esc(tr.get("train_images")),
                    esc(tr.get("config")), esc(tr.get("remote_log")), esc(d.get("checkpoint")), esc(e.get("eval_cmd")), esc(e.get("remote_log")),
                    f4(e["peak"]["f1"]), f4(e["peak"].get("precision")), f4(e["peak"].get("recall")), f4(e["pixel"]["iou"]),
                    f4(e["pixel"].get("f1")), esc(e.get("n_images")),
                    ("<tr><th>based on</th><td>%s</td></tr>" % esc(", ".join(map(str, d.get("based_on", []))))) if d.get("based_on") else ""))
        if d.get("notes"):
            H.append("<p><small>%s</small></p>" % esc(d["notes"]))
        H.append('<div class="plots">')
        for p in d.get("proof", [])[:1]:
            H.append(fig(imgs, p, "<b>%s</b> test prediction overlay (%s)" % (m, esc(os.path.basename(p)))))
        for f in ev.get("figures", []):
            if f.get("section") == m:
                H.append(fig(imgs, f["file"], f["caption"]))
        H.append("</div></div>")
    H.append("</section>")

    # cluster / logs evidence
    other = [f for f in ev.get("figures", []) if f.get("section") in ("cluster", "curves", "logs", "reeval")]
    if other:
        H.append('<section><h2>S3DF evidence: jobs, training curves, logs, independent read-back</h2><div class="plots">')
        for f in other:
            H.append(fig(imgs, f["file"], f["caption"]))
        H.append("</div></section>")

    # detection
    if det:
        e = det["eval"]
        H.append('<section><h2>Detection (low priority): <code>detection</code></h2><p>%s. Labels: %s. Test mAP50 <b>%s</b>, precision %s, recall %s on %s images. Checkpoint <code>%s</code>, eval log <code>%s</code>.</p><div class="plots">'
                 % (esc(det.get("model")), esc(det.get("labels_from")), f4(e.get("map50")), f4(e.get("precision")), f4(e.get("recall")),
                    esc(e.get("n_images")), esc(det.get("checkpoint")), esc(e.get("remote_log"))))
        for p in det.get("proof", [])[:2]:
            H.append(fig(imgs, p, "<b>detection</b> test boxes (%s)" % esc(os.path.basename(p))))
        H.append("</div></section>")

    # findings / not covered / rerun / audit
    if ev.get("findings"):
        H.append('<section><h2>Findings</h2><ul class="findings">%s</ul></section>' % "".join("<li>%s</li>" % f for f in ev["findings"]))
    nc = list(ev.get("not_covered", [])) + ["<code>%s</code>: no result on main when this page was built." % m for m in MODELS if not res[m]]
    if nc:
        H.append('<section><h2>Not covered</h2><ul class="findings">%s</ul></section>' % "".join("<li>%s</li>" % f for f in nc))
    cmds = ev.get("commands") or []
    cmds += ["# %s\n%s" % (m, res[m]["eval"]["eval_cmd"]) for m in landed]
    H.append("<section><h2>Run it again (on S3DF, from the repo root)</h2><pre>%s</pre></section>" % esc("\n\n".join(cmds)))
    H.append("<details><summary>Audit trail</summary><div>%s<p class=muted>Page assembled by <code>scripts/make_proof_report.py</code> from the result.json files on main and "
             "<code>results/proof_report/evidence.json</code>; all images are PNGs produced on S3DF (matplotlib, Agg) and copied byte-for-byte.</p></div></details>"
             % "".join("<p>%s</p>" % x for x in ev.get("audit", [])))
    H.append("</div></body></html>")
    page = "\n".join(H)
    os.makedirs(OUT_DIR, exist_ok=True)
    open(os.path.join(OUT_DIR, "index.html"), "w").write(page)
    n_img = page.count("<img ")
    rj = {"statement": "proof_report", "page": "results/proof_report/index.html", "made_with": "/proof-report skill conventions; scripts/make_proof_report.py",
          "covers": MODELS + (["detection"] if det else []), "models_with_results": landed, "n_img": n_img,
          "evidence": "results/proof_report/evidence.json", "built": now, "main_commit": a.commit,
          "notes": "All screenshots are S3DF artefacts (seg/viz.py overlays, scripts/proof_figs.py renders of squeue/logs/curves, independent re-evaluation)."}
    json.dump(rj, open(os.path.join(OUT_DIR, "result.json"), "w"), indent=1)
    print("wrote %s: %.2f MB, %d <img>, models with results: %s" % (os.path.join(OUT_DIR, "index.html"), len(page) / 2 ** 20, n_img, landed))


if __name__ == "__main__":
    main()

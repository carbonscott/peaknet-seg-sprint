"""Frozen merge check. usage: python checks/pkcheck.py <statement>   (run in the repo root)

Exit 0 = the statement is done. Prints the reason for any failure. See STATEMENTS.md.
"""
import glob
import importlib.util
import json
import os
import re
import sys

MODEL_STATEMENTS = ["peaknet_baseline", "hf_segformer", "hf_upernet", "lucid_segformer",
                    "lucid_unet", "best_model"]
UPSTREAM_OK = re.compile(r"(huggingface\.co/|github\.com/lucidrains/|github\.com/carbonscott/peaknet"
                         r"|github\.com/huggingface/)")
MAX_FILE = 3 * 1024 * 1024


class Fail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


def load(name):
    p = "results/%s/result.json" % name
    need(os.path.isfile(p), "missing %s" % p)
    try:
        d = json.load(open(p))
    except Exception as e:
        raise Fail("%s is not valid JSON: %s" % (p, e))
    need(isinstance(d, dict), "%s must hold a JSON object" % p)
    need(d.get("statement") == name, '%s: "statement" must be "%s"' % (p, name))
    return d


def unit(x, what):
    need(isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= x <= 1.0,
         "%s must be a number in [0,1], got %r" % (what, x))


def png_list(name, d, minimum=1):
    pr = d.get("proof")
    need(isinstance(pr, list) and len(pr) >= minimum,
         '"proof" must list at least %d PNG file(s) under results/%s/' % (minimum, name))
    for f in pr:
        need(isinstance(f, str) and f.startswith("results/%s/" % name) and f.endswith(".png"),
             "proof entry %r must be a .png path under results/%s/" % (f, name))
        need(os.path.isfile(f), "proof file %s does not exist" % f)
        need(os.path.getsize(f) <= MAX_FILE, "proof file %s is larger than 3 MB" % f)
        need(open(f, "rb").read(8) == b"\x89PNG\r\n\x1a\n", "%s is not a PNG" % f)


def no_big_files(name):
    for f in glob.glob("results/%s/**" % name, recursive=True):
        if os.path.isfile(f):
            need(os.path.getsize(f) <= MAX_FILE or f.endswith(".html") and os.path.getsize(f) <= 15 * 2**20,
                 "%s is too large (3 MB limit; 15 MB for .html). Keep checkpoints on S3DF." % f)


def split_info():
    d = load("split")
    return d


def check_split():
    d = load("split")
    st = d.get("stores")
    need(isinstance(st, dict) and set(st) == {"train", "val", "test"},
         '"stores" must be an object with exactly train, val, test')
    runs = {}
    allst = []
    for k, v in st.items():
        need(isinstance(v, list) and v and all(isinstance(s, str) and s.endswith(".zarr") for s in v),
             '"stores.%s" must be a non-empty list of .zarr basenames' % k)
        need(all("/" not in s for s in v), "store names must be basenames, not paths")
        for s in v:
            m = re.match(r"(.+_peaknet)\.\d{4}\.zarr$", s)
            need(m and ".v3." not in s, "store %s is not a v2 {exp}_{run}_peaknet.NNNN.zarr name" % s)
            runs.setdefault(m.group(1), set()).add(k)
        allst += v
    need(len(allst) == len(set(allst)), "a store appears twice")
    need(len(allst) >= 450, "the split must cover at least 450 of the 504 v2 stores (has %d)" % len(allst))
    leak = [r for r, ks in runs.items() if len(ks) > 1]
    need(not leak, "runs split across sets (leakage): %s" % ", ".join(sorted(leak)[:5]))
    n = d.get("n_images")
    need(isinstance(n, dict) and all(isinstance(n.get(k), int) and n[k] > 0 for k in st),
         '"n_images" must give a positive int for train, val, test')
    need(n["test"] >= 1000, "the test set must hold at least 1000 images")
    need(isinstance(d.get("label_values"), list) and d["label_values"], '"label_values" must list the label ids seen')
    need(isinstance(d.get("peak_label"), int), '"peak_label" must be the int label of the peak class')
    need(os.path.isfile("seg/data.py"), "seg/data.py (the shared loader) must exist")
    need(isinstance(d.get("remote_manifest"), str) and d["remote_manifest"].startswith("/sdf/"),
         '"remote_manifest" must be the absolute S3DF path of the split manifest')


def load_eval_module():
    p = "seg/evaluate.py"
    need(os.path.isfile(p), "seg/evaluate.py is missing")
    sys.path.insert(0, os.getcwd())
    spec = importlib.util.spec_from_file_location("seg_evaluate_under_check", p)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:
        raise Fail("importing seg/evaluate.py failed (only numpy and scipy are available to the check): %r" % e)
    for fn in ("pixel_metrics", "peak_metrics"):
        need(callable(getattr(m, fn, None)), "seg/evaluate.py must define %s" % fn)
    return m


def close(a, b, what):
    need(isinstance(a, (int, float)) and abs(a - b) < 1e-6, "%s: expected %s, got %r" % (what, b, a))


def blob(img, y, x, r=1, v=1):
    img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = v


def check_evalkit():
    import numpy as np
    load("evalkit")
    m = load_eval_module()
    P, K = m.pixel_metrics, m.peak_metrics
    t = np.zeros((64, 64), np.int64)
    for y, x in [(10, 10), (30, 40), (50, 20)]:
        blob(t, y, x)
    # identical masks
    r = P(t, t)
    for k in ("iou", "f1", "precision", "recall"):
        close(r[k], 1.0, "pixel_metrics(identical).%s" % k)
    r = K(t, t)
    for k in ("f1", "precision", "recall"):
        close(r[k], 1.0, "peak_metrics(identical).%s" % k)
    need(r.get("tp") == 3 and r.get("fp") == 0 and r.get("fn") == 0, "peak_metrics(identical) counts: %r" % r)
    # empty prediction
    e = np.zeros_like(t)
    r = P(e, t)
    close(r["recall"], 0.0, "pixel_metrics(empty pred).recall")
    close(r["precision"], 1.0, "pixel_metrics(empty pred).precision (0/0 is 1.0)")
    close(r["f1"], 0.0, "pixel_metrics(empty pred).f1")
    close(r["iou"], 0.0, "pixel_metrics(empty pred).iou")
    r = P(e, e)
    close(r["iou"], 1.0, "pixel_metrics(both empty).iou (0/0 is 1.0)")
    # partial pixel overlap: pred covers 9 target px + 9 extra px
    p = np.zeros_like(t)
    blob(p, 10, 10)
    blob(p, 60, 60)
    r = P(p, t)
    need(r.get("tp") == 9 and r.get("fp") == 9 and r.get("fn") == 18, "pixel_metrics counts: %r" % r)
    close(r["iou"], 9 / 36, "pixel_metrics.iou")
    close(r["precision"], 0.5, "pixel_metrics.precision")
    close(r["recall"], 9 / 27, "pixel_metrics.recall")
    # peaks: two shifted by 2 px (match at tol=3), one spurious far away, one missed
    p = np.zeros_like(t)
    blob(p, 12, 10)
    blob(p, 30, 42)
    blob(p, 5, 60)
    r = K(p, t, tol=3.0)
    need(r.get("tp") == 2 and r.get("fp") == 1 and r.get("fn") == 1, "peak_metrics counts at tol=3: %r" % r)
    close(r["precision"], 2 / 3, "peak_metrics.precision")
    close(r["recall"], 2 / 3, "peak_metrics.recall")
    r = K(p, t, tol=1.0)
    need(r.get("tp") == 0, "peak_metrics at tol=1 must not match peaks 2 px apart: %r" % r)
    # one-to-one matching: one prediction between two close targets
    t2 = np.zeros((32, 32), np.int64)
    t2[10, 10] = 1
    t2[10, 14] = 1
    p2 = np.zeros_like(t2)
    p2[10, 12] = 1
    r = K(p2, t2, tol=3.0)
    need(r.get("tp") == 1 and r.get("fn") == 1 and r.get("fp") == 0, "matching must be one-to-one: %r" % r)
    # 8-connectivity: a diagonal pair is one peak
    t3 = np.zeros((16, 16), np.int64)
    t3[5, 5] = t3[6, 6] = 1
    r = K(t3, t3)
    need(r.get("tp") == 1, "diagonal neighbours must form one peak (8-connectivity): %r" % r)
    # batches: (B,H,W) counts are summed over the batch (micro average)
    b_t = np.stack([t, t])
    b_p = np.stack([t, e])
    r = K(b_p, b_t)
    need(r.get("tp") == 3 and r.get("fn") == 3, "batched peak_metrics must sum counts over the batch: %r" % r)
    r = P(b_p, b_t)
    need(r.get("tp") == 27 and r.get("fn") == 27, "batched pixel_metrics must sum counts: %r" % r)
    # other label ids: cls selects the class
    t4 = t * 2
    r = P(t4, t4, cls=2)
    close(r["iou"], 1.0, "pixel_metrics(cls=2)")
    r = P(t4, t4, cls=1)
    close(r["iou"], 1.0, "pixel_metrics(cls=1) on masks without class 1 (both empty)")


def check_model(name):
    d = load(name)
    s = split_info()
    need(isinstance(d.get("model"), str) and d["model"], '"model" must name the model')
    up = d.get("upstream")
    need(isinstance(up, list) and up, '"upstream" must list the projects reused')
    for u in up:
        need(isinstance(u, dict) and isinstance(u.get("url"), str) and isinstance(u.get("ref"), str) and u["ref"],
             'each "upstream" entry needs "url" and "ref" (version, tag or commit)')
    if name != "best_model":
        need(any(UPSTREAM_OK.search(u["url"]) for u in up),
             '"upstream" must include a Hugging Face, lucidrains or carbonscott/peaknet URL')
    code = d.get("code")
    need(isinstance(code, list) and code, '"code" must list the repo files this model uses')
    for f in code:
        need(isinstance(f, str) and os.path.isfile(f), "code file %r is not in the repository" % f)
    tr = d.get("train")
    need(isinstance(tr, dict), '"train" must be an object')
    for k in ("slurm_jobid", "remote_log", "config"):
        need(isinstance(tr.get(k), str) and tr[k], '"train.%s" must be a non-empty string' % k)
    need(tr["remote_log"].startswith("/sdf/"), '"train.remote_log" must be an absolute S3DF path')
    need(isinstance(tr.get("gpus"), int) and tr["gpus"] >= 1, '"train.gpus" must be an int >= 1')
    need(isinstance(tr.get("wallclock_min"), (int, float)) and tr["wallclock_min"] > 0, '"train.wallclock_min" > 0')
    need(isinstance(tr.get("train_images"), int) and tr["train_images"] > 0, '"train.train_images" must be > 0')
    need(isinstance(d.get("checkpoint"), str) and d["checkpoint"].startswith("/sdf/"),
         '"checkpoint" must be an absolute S3DF path')
    ev = d.get("eval")
    need(isinstance(ev, dict), '"eval" must be an object')
    need(ev.get("split") == "test", '"eval.split" must be "test"')
    need(ev.get("n_images") == s["n_images"]["test"],
         '"eval.n_images" (%r) must equal the split\'s test size %r: evaluate on the whole test set'
         % (ev.get("n_images"), s["n_images"]["test"]))
    need(isinstance(ev.get("eval_cmd"), str) and "evaluate" in ev["eval_cmd"],
         '"eval.eval_cmd" must be the command that ran seg/evaluate.py')
    need(isinstance(ev.get("remote_log"), str) and ev["remote_log"].startswith("/sdf/"),
         '"eval.remote_log" must be an absolute S3DF path')
    for grp, keys in (("pixel", ("iou", "f1", "precision", "recall")), ("peak", ("f1", "precision", "recall"))):
        g = ev.get(grp)
        need(isinstance(g, dict), '"eval.%s" must be an object' % grp)
        for k in keys:
            unit(g.get(k), "eval.%s.%s" % (grp, k))
    need(isinstance(ev.get("peak_tol_px"), (int, float)) and ev["peak_tol_px"] == 3,
         '"eval.peak_tol_px" must be 3 (the team-wide peak matching tolerance)')
    png_list(name, d)
    no_big_files(name)
    if name == "best_model":
        need(isinstance(d.get("based_on"), list) and d["based_on"], '"based_on" must list what was combined')
        mine = ev["peak"]["f1"]
        for o in MODEL_STATEMENTS:
            if o != name and os.path.isfile("results/%s/result.json" % o):
                try:
                    of = json.load(open("results/%s/result.json" % o))["eval"]["peak"]["f1"]
                except Exception:
                    continue
                need(mine >= of, "best_model peak F1 %.4f is below %s (%.4f)" % (mine, o, of))


def model_rows():
    rows = []
    for o in MODEL_STATEMENTS:
        p = "results/%s/result.json" % o
        if os.path.isfile(p):
            try:
                d = json.load(open(p))
                rows.append((o, d["eval"]["peak"]["f1"], d["eval"]["pixel"]["iou"]))
            except Exception:
                pass
    return rows


def check_leaderboard():
    d = load("leaderboard")
    rows = model_rows()
    need(len(rows) >= 3, "the leaderboard needs at least 3 model statements with results on main")
    ent = d.get("entries")
    need(isinstance(ent, list), '"entries" must be a list')
    have = {e.get("statement"): e for e in ent if isinstance(e, dict)}
    for o, f1, iou in rows:
        need(o in have, "leaderboard is missing %s" % o)
        close(have[o].get("peak_f1"), f1, "leaderboard %s.peak_f1" % o)
        close(have[o].get("pixel_iou"), iou, "leaderboard %s.pixel_iou" % o)
    f1s = [e.get("peak_f1") for e in ent]
    need(f1s == sorted(f1s, reverse=True), "entries must be sorted by peak_f1, best first")
    need(os.path.isfile("README.md"), "README.md is missing")
    readme = open("README.md").read()
    for o, _, _ in rows:
        need(o in readme, "README.md leaderboard table must mention %s" % o)
    no_big_files("leaderboard")


def check_proof_report():
    load("proof_report")
    p = "results/proof_report/index.html"
    need(os.path.isfile(p), "missing %s" % p)
    need(os.path.getsize(p) <= 15 * 2**20, "%s is larger than 15 MB" % p)
    h = open(p, encoding="utf-8", errors="replace").read()
    imgs = re.findall(r"<img\b[^>]*>", h)
    need(len(imgs) >= 5, "the report needs at least 5 screenshots (<img>), has %d" % len(imgs))
    for o, _, _ in model_rows():
        need(o in h, "the report must cover %s" % o)
    no_big_files("proof_report")


def check_detection():
    d = load("detection")
    s = split_info()
    for k in ("model", "labels_from"):
        need(isinstance(d.get(k), str) and d[k], '"%s" must be a non-empty string' % k)
    up = d.get("upstream")
    need(isinstance(up, list) and up and all(isinstance(u, dict) and u.get("url") and u.get("ref") for u in up),
         '"upstream" must list projects reused, each with url and ref')
    need(isinstance(d.get("checkpoint"), str) and d["checkpoint"].startswith("/sdf/"), '"checkpoint" must be an S3DF path')
    ev = d.get("eval")
    need(isinstance(ev, dict) and ev.get("split") == "test", '"eval.split" must be "test"')
    need(ev.get("n_images") == s["n_images"]["test"], '"eval.n_images" must equal the test size')
    for k in ("map50", "precision", "recall"):
        unit(ev.get(k), "eval.%s" % k)
    need(isinstance(ev.get("remote_log"), str) and ev["remote_log"].startswith("/sdf/"), '"eval.remote_log" must be an S3DF path')
    png_list("detection", d)
    no_big_files("detection")


def main():
    name = sys.argv[1]
    try:
        if name == "split":
            check_split()
        elif name == "evalkit":
            check_evalkit()
        elif name in MODEL_STATEMENTS:
            check_model(name)
        elif name == "leaderboard":
            check_leaderboard()
        elif name == "proof_report":
            check_proof_report()
        elif name == "detection":
            check_detection()
        else:
            raise Fail("unknown statement %s" % name)
    except Fail as e:
        print("FAIL %s: %s" % (name, e))
        sys.exit(1)
    print("PASS %s" % name)


if __name__ == "__main__":
    main()

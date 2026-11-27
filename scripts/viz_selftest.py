"""Self-test for seg/viz.py on a real store: synthetic prediction with known errors.
usage: python scripts/viz_selftest.py <store path> <index> <out dir>"""
import sys, json, numpy as np, zarr
from scipy.optimize import linear_sum_assignment
from seg.viz import save_overlay, save_overlay_grid, match_peaks, centroids

def hungarian_tp(p, g, tol):
    pc, gc = centroids(p), centroids(g)
    if not len(pc) or not len(gc):
        return 0
    d = np.linalg.norm(pc[:, None] - gc[None], axis=-1)
    c = np.where(d <= tol, d, 1e6)
    r, k = linear_sum_assignment(c)
    return int((d[r, k] <= tol).sum())

store, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
z = zarr.open(store, "r")
samples = []
rng = np.random.default_rng(0)
for i in (idx, idx + 1, idx + 2):
    img, lab = z["images"][i], z["labels"][i]
    gt = lab == 1
    # synthetic prediction: shift the mask 2 px, drop ~25% of components, add 5 spurious blobs
    from scipy import ndimage
    L, n = ndimage.label(gt, structure=np.ones((3, 3)))
    keep = rng.random(n) > 0.25
    pred = np.isin(L, np.flatnonzero(keep) + 1)
    pred = np.roll(pred, 2, axis=1)
    for _ in range(5):
        y, x = rng.integers(50, img.shape[0] - 50, 2)
        pred[y:y + 2, x:x + 2] = True
    _, _, pairs = match_peaks(pred, gt, 3.0)
    ht = hungarian_tp(pred, gt, 3.0)
    print(json.dumps({"i": i, "n_gt": int(n), "kept": int(keep.sum()), "match_tp": len(pairs), "hungarian_tp": ht}))
    assert len(pairs) == ht
    samples.append((img, lab, pred.astype(np.int64), "%s[%d]" % (store.rstrip("/").split("/")[-1], i)))
m = save_overlay(*samples[0][:3], out + "/viz_selftest_single.png", title="viz self-test (synthetic pred) " + samples[0][3], extra="synthetic: 2px shift, 25% dropped, 5 spurious")
print(json.dumps(m))
print(json.dumps(save_overlay_grid(samples, out + "/viz_selftest_grid.png", title="viz self-test grid (synthetic pred)")[1]))

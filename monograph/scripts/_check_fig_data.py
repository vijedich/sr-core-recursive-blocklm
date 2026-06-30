import json, os, sys

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval", "entmin")

def load(name):
    return json.load(open(os.path.join(DATA, name)))

def code_ratio_ho(metrics):
    lf = metrics["Lfin_heldout"]
    others = [v for k,v in lf.items() if k != "code"]
    return lf["code"] / (sum(others) / len(others))

configs = [
    (0.000, "eval_compare_entmin.json",       "a"),
    (0.001, "eval_compare_entmin.json",       "b"),
    (0.003, "eval_compare_entmin_lam003.json","b"),
    (0.005, "eval_compare_entmin_lam005.json","b"),
    (0.007, "eval_compare_entmin_lam007.json","b"),
]
print("=== Router consolidation ===")
for lam, fname, side in configs:
    m = load(fname)["metrics_" + side]
    print("lam=%.3f  ent=%.4f  ucores=%d  K16=%.1f  K24=%.1f  hardov=%.4f" % (
        lam, m["router_entropy"], m["unique_cores"],
        m["lb"]["16"]["lru"], m["lb"]["24"]["lru"], m["hard_overlap_eval"]))

print()
print("=== Pareto points ===")
configs2 = [
    ("ctrl",  "eval_compare_entmin_lam003.json", "a", "eval_quality_lam003.json", "a"),
    ("lam003","eval_compare_entmin_lam003.json", "b", "eval_quality_lam003.json", "b"),
    ("lam005","eval_compare_entmin_lam005.json", "b", "eval_quality_lam005.json", "b"),
    ("lam007","eval_compare_entmin_lam007.json", "b", "eval_quality_lam007.json", "b"),
    ("H375",  "eval_compare_target_H375.json",   "b", "eval_quality_H375.json",   "b"),
    ("H370",  "eval_compare_target_H370.json",   "b", "eval_quality_H370.json",   "b"),
]
for label, ec, es, eq, qs in configs2:
    K24 = load(ec)["metrics_" + es]["lb"]["24"]["lru"]
    cr  = code_ratio_ho(load(eq)["metrics_" + qs])
    print("%-8s  K24=%.1f  code_ratio_HO=%.4f" % (label, K24, cr))

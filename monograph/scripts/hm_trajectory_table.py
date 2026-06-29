"""Aggregiert die per-Config Trajectory-JSONs zur Verlaufstabelle.

  step | model | Lfin | WS | reuse_p90 | K8 KB/token | rel_dense | rel_naked | anytime | dead

rel_dense = K8(model) / K8(dense_d24, gleicher step)  [dense K8 ~konstant]
rel_naked = K8(model) / K8(naked_b32_R6, gleicher step)
"""
import glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def load():
    data = {}
    for p in glob.glob(os.path.join(RES, "hm_traj_*_s0.json")):
        d = json.load(open(p, encoding="utf-8"))
        data[d["experiment"]] = {r["step"]: r for r in d["rows"]}
    return data


def main():
    data = load()
    if not data:
        print("Noch keine Trajectory-Daten."); return
    dense = data.get("hm_dense_d24", {})
    naked = data.get("hm_naked_b32_R6", {})
    steps = sorted({s for m in data.values() for s in m})
    order = ["hm_srcore_b32_R6", "hm_naked_b32_R6", "hm_dense_d24",
             "hm_srcore_b64_R6", "hm_srcore_b32_R2"]
    order = [e for e in order if e in data]
    hdr = (f'{"step":>6} {"model":18} {"Lfin":7} {"WS":5} {"reuP90":6} '
           f'{"K8_KB":8} {"relDense":8} {"relNaked":8} {"anyt":6} {"dead":4}')
    print(hdr); print("-" * len(hdr))
    for step in steps:
        for e in order:
            r = data[e].get(step)
            if not r:
                continue
            k8 = r["k8_kb_per_token"]
            rd = dense.get(step, {}).get("k8_kb_per_token")
            rn = naked.get(step, {}).get("k8_kb_per_token")
            reld = f'{k8/rd:.3f}' if rd else "-"
            reln = f'{k8/rn:.3f}' if rn else "-"
            print(f'{step:>6} {e:18} {r["Lfin"]:<7.3f} {r["WS"]:<5.1f} {r["reuse_p90"]:<6.0f} '
                  f'{k8:<8.1f} {reld:8} {reln:8} {r["anytime"]:<6.3f} {r["dead_blocks"]:<4}')
        print()


if __name__ == "__main__":
    main()

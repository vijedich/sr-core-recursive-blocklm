import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rblm import model_io

ck = "results/hm_srcore_b64_k8_R6_s0_10k.pt"
model, arch, step = model_io.load_checkpoint(ck, device="cpu")
print(model_io.label(arch, step))
print("n_blocks=%d  k=%d  R=%d  core_mode=%s" % (
    arch["n_blocks"], arch["k"], arch["R"], arch["core_mode"]))
print("params=%.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6,))
print("OK")

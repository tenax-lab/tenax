"""S5 at chi=32 (user choice 2026-11-01, option 1 of the chi=16 layout diagnosis).

_s5_config(3, 32, 200) unchanged except ctm.max_iter 1000 -> 2000,
gs_log_probes=True, checkpoint every step.  Fresh run (no resume).
Start: params of the s5probes step-20 checkpoint (= start of chi=16 step 21,
E=-0.5735019318 at chi=16).  Seed env: a cold chi=32 CTM (1751 sweeps,
converged, layout certified, layout 4935970a5ef3, E=-0.5723747465) at the
chi=16 step-21 accepted point, alpha=2.294e-3 along the step-21 direction.
Code: branch run/s5-probes @ 824bd683.
Usage: python run_chi32.py OUT_DIR INPUTS_DIR REPO
"""
import dataclasses, pickle, sys, time
from pathlib import Path

OUT, INP, REPO = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, REPO)
import benchmarks.symmetric_ad.pipeline as pl  # noqa: E402
import jax  # noqa: E402
from tenax.algorithms.ipeps_optimize import _layout_key  # noqa: E402

jax.config.update("jax_compilation_cache_dir", REPO + "/.jax_cache_s5chi32")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
OUT.mkdir(parents=True, exist_ok=True)
ck = pickle.load(open(INP / "s5probes_step20.pkl", "rb"))
envs = pickle.load(open(INP / "env_cold32.pkl", "rb"))
base = pl._s5_config(3, 32, 200)
cfg = dataclasses.replace(base, ctm=dataclasses.replace(base.ctm, max_iter=2000),
                          gs_checkpoint_path=str(OUT / "ckpt"), gs_checkpoint_every=1,
                          gs_log_probes=True)
print("devices", jax.devices(), flush=True)
print("ckpt step", ck["step"], "seed layout", _layout_key(envs), flush=True)
print("chi", cfg.ctm.chi, "ctm max_iter", cfg.ctm.max_iter, "probe_max_iter", cfg.ctm.probe_max_iter,
      "conv_tol", cfg.ctm.conv_tol, "grad tol", cfg.gs_grad_norm_tol,
      "layout_recheck", cfg.gs_layout_recheck, "reject_rise", cfg.gs_reject_energy_rise,
      "log_probes", cfg.gs_log_probes, "resume", cfg.gs_resume, flush=True)
t0 = time.time()
out, failed = pl._optimize_or_fail(pl.xxz_gate(0.3), tuple(ck["params"]), cfg, envs, 1, [])
with open(OUT / "s5_end.pkl", "wb") as f:
    pickle.dump({"out": out, "failed": failed}, f)
print("S5 chi32 done", "failed" if failed else "ok", f"wall {time.time()-t0:.0f}s", flush=True)
if failed:
    print(failed[1].get("error"), flush=True)

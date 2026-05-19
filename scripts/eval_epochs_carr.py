"""Evaluate multiple checkpoints against Carr — one line per checkpoint."""

import sys, os, glob
import numpy as np, torch, xarray as xr, datetime as dt
from scipy.ndimage import sobel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "zeus"))

from stereo_winds.config import GOES16_CONFIG, GOES18_CONFIG
from stereo_winds.remap import load_remap_lut, remap_image
from stereo_winds.solver import build_design_matrix, solve_stereo_winds, pixels_to_wind_ms
from stereo_winds.data_loading import load_native_abi
from stereo_winds.disparity import StereoDisparity
from stereo_winds.time_model import compute_scene_times
from stereo_winds.navigation import geodetic_to_fixed_grid, scanning_angle_to_pixel
from stereo_winds.validation.metrics import rmsvd, speed_bias, height_rmse, correlation

cache_root = "/home/ubuntu/earthnet-us-east-3/cache"
T0 = dt.datetime(2025, 1, 8, 19, 0)
DT = 10.0
sat_a, sat_b = GOES16_CONFIG, GOES18_CONFIG

# Load shared data once
col_b, row_b = load_remap_lut(os.path.join(cache_root, "remap_lut_goes16_goes18.npz"))
pdata = np.load(os.path.join(cache_root, "parallax_goes16_goes18.npz"))
w_u, w_v = pdata["w_u"], pdata["w_v"]
st = compute_scene_times(T0, DT, sat_a, sat_b)
H_matrix = build_design_matrix(
    w_u, w_v, dt_a_minus=st["A_minus"], dt_a_plus=st["A_plus"],
    dt_b_minus=st["B_minus"], dt_b_plus=st["B_plus"],
)

def find_abi(s, b, t):
    y, d, h = t.year, int(t.strftime("%j")), t.hour
    return glob.glob(os.path.join(
        cache_root, s, "ABI", str(y), f"{d:03d}", f"{h:02d}",
        f"*{b}*s{y}{d:03d}{t.strftime('%H%M')}*.nc"))[0]

tm = T0 - dt.timedelta(minutes=DT)
tp = T0 + dt.timedelta(minutes=DT)
am, _ = load_native_abi(find_abi("goes16", "C14", tm), satellite_id="goes16")
a0, _ = load_native_abi(find_abi("goes16", "C14", T0), satellite_id="goes16")
ap, _ = load_native_abi(find_abi("goes16", "C14", tp), satellite_id="goes16")
bm = remap_image(load_native_abi(find_abi("goes18", "C14", tm), satellite_id="goes18")[0], col_b, row_b)
bp = remap_image(load_native_abi(find_abi("goes18", "C14", tp), satellite_id="goes18")[0], col_b, row_b)
valid = np.isfinite(a0) & np.isfinite(am) & np.isfinite(ap) & np.isfinite(bm) & np.isfinite(bp)
images = {"A_minus": am, "A0": a0, "A_plus": ap, "B_minus": bm, "B_plus": bp}

# Load Carr
ds = xr.open_dataset(os.path.join(BASE, "carr_data/GOES_GOES_B14_20250081900208.nc"), engine="h5netcdf")
cl, clo = ds["lat"].values, ds["lon"].values
cu, cv = ds["V_3D"].values[:, 0], ds["V_3D"].values[:, 1]
ch, cdqf = ds["H_3D"].values, ds["DQF_3D"].values
ds.close()
cg = (cdqf == 0) & np.isfinite(ch) & np.isfinite(cu) & np.isfinite(cv) & (ch >= 0) & (ch <= 20000)
QA = dict(chi2_max=0.2, sigma_h_max=5000., h_grad_max=3000., wind_speed_max=100., w_mag_min=0.0003)

TILE_SIZE = int(os.environ.get("TILE_SIZE", "512"))
TILE_OVERLAP = TILE_SIZE // 2

H_BINS = [(0, 3000), (3000, 6000), (6000, 12000), (12000, 20000)]
P_BINS = [(0, 1), (1, 3), (3, 6), (6, 12), (12, 99)]

# Pre-compute Carr collocation pixels using Carr's reported height (parallax-correct)
g = cg
xa_pre, ya_pre = geodetic_to_fixed_grid(cl[g], clo[g], sat_a, h_m=ch[g])
cf_pre, rf_pre = scanning_angle_to_pixel(xa_pre, ya_pre, sat_a)
ci_pre = np.clip(np.round(cf_pre).astype(int), 0, sat_a.n_cols - 1)
ri_pre = np.clip(np.round(rf_pre).astype(int), 0, sat_a.n_rows - 1)
ib_pre = (cf_pre >= 0) & (cf_pre < sat_a.n_cols) & (rf_pre >= 0) & (rf_pre < sat_a.n_rows) & np.isfinite(cf_pre)
chh_g, cuh_g, cvh_g = ch[g], cu[g], cv[g]


def evalck(ckpt_path, label):
    disp = StereoDisparity(model_ckpt_path=ckpt_path, tile_size=TILE_SIZE, overlap=TILE_OVERLAP, batch_size=8, device="cuda")
    flows = disp.compute_all(images)
    for k in flows:
        flows[k][:, ~valid] = np.nan
    sol = solve_stereo_winds(flows, H_matrix, sat_a=sat_a, sat_b=sat_b, n_iter=3)
    u_ms, v_ms = pixels_to_wind_ms(sol["V_u"], sol["V_v"], sat_a, dt_seconds=1.0)
    h = sol["h"]
    qf = sol["quality_flag"].copy()
    qf[~valid] = 0.0
    hf = np.where(np.isfinite(h), h, 0.)
    grad = np.sqrt(sobel(hf, axis=1)**2 + sobel(hf, axis=0)**2) / 8.
    spd = np.sqrt(u_ms**2 + v_ms**2)
    wm = np.sqrt(w_u**2 + w_v**2)
    qa = ((qf > 0) & np.isfinite(h) & np.isfinite(sol["chi2"]) & (sol["chi2"] <= QA["chi2_max"])
          & np.isfinite(sol["sigma_h"]) & (sol["sigma_h"] <= QA["sigma_h_max"])
          & (grad <= QA["h_grad_max"]) & np.isfinite(spd) & (spd <= QA["wind_speed_max"])
          & (wm >= QA["w_mag_min"]))

    D1, D2, D3, D4 = flows["D1"], flows["D2"], flows["D3"], flows["D4"]
    emp_par_u = ((D3[0] + D4[0]) / 2) - ((D1[0] + D2[0]) / 2)
    emp_par_v = ((D3[1] + D4[1]) / 2) - ((D1[1] + D2[1]) / 2)
    emp_par = np.sqrt(emp_par_u**2 + emp_par_v**2)

    ok = ib_pre & (qf[ri_pre, ci_pre] > 0) & np.isfinite(h[ri_pre, ci_pre]) & np.isfinite(u_ms[ri_pre, ci_pre]) & qa[ri_pre, ci_pre]

    def _row(mask, prefix):
        if mask.sum() < 30:
            return f"  {prefix}  N={mask.sum():>5}  (too few)"
        ah = h[ri_pre[mask], ci_pre[mask]]
        au = u_ms[ri_pre[mask], ci_pre[mask]]
        av = v_ms[ri_pre[mask], ci_pre[mask]]
        chm, cum, cvm = chh_g[mask], cuh_g[mask], cvh_g[mask]
        hr = height_rmse(ah, chm)
        hb = float(np.mean(ah - chm))
        rv = rmsvd(au, av, cum, cvm)
        sb = speed_bias(au, av, cum, cvm)
        return f"  {prefix}  N={mask.sum():>6,}  H_RMSE={hr:>5.0f}m  H_bias={hb:>+6.0f}m  RMSVD={rv:>5.2f}  SpBias={sb:>+5.2f}"

    print(f"\n{label}")
    print(_row(ok, "all     "))
    for lo, hi in H_BINS:
        m = ok & (chh_g >= lo) & (chh_g < hi)
        print(_row(m, f"h={lo/1000:>2.0f}-{hi/1000:>2.0f}km"))

    # Empirical parallax distribution at high-cloud subset (where stereo matters most)
    hi_subset = ok & (chh_g >= 6000) & (chh_g < 12000)
    if hi_subset.sum() > 0:
        ep_pts = emp_par[ri_pre, ci_pre]
        bin_counts = []
        for lo, hi in P_BINS:
            bin_counts.append(((ep_pts >= lo) & (ep_pts < hi) & hi_subset).sum())
        print(f"  6-12km |p| distribution:  " + "  ".join(
            f"{lo}-{hi}px:{c}" for (lo, hi), c in zip(P_BINS, bin_counts)))


# Pretrained baseline
evalck(os.path.join(BASE, "zeus/zeus/networks/weights/raft-128.202509.epoch1434.ckpt"), "Pretrained")

# Find per-epoch checkpoints from the latest run
ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else None
if ckpt_dir is None:
    # Auto-find latest run's checkpoint dir
    dirs = sorted(glob.glob(os.path.join(BASE, "output/training/stereo-winds-v2/*/checkpoints")),
                  key=os.path.getmtime)
    if not dirs:
        dirs = sorted(glob.glob(os.path.join(BASE, "stereo-winds-v2/*/checkpoints")),
                      key=os.path.getmtime)
    if dirs:
        ckpt_dir = dirs[-1]

if ckpt_dir:
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    for ckpt in ckpts:
        name = os.path.basename(ckpt).replace(".ckpt", "")
        evalck(ckpt, name)
else:
    print("No checkpoint directory found. Pass path as argument.")

print("-" * 100)

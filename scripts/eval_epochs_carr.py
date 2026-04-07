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

print(f"{'Label':45s}  {'N':>6s}  {'H_RMSE':>7s}  {'H_bias':>7s}  {'H_corr':>6s}  {'RMSVD':>7s}  {'SpBias':>7s}")
print("-" * 100)


def evalck(ckpt_path, label):
    disp = StereoDisparity(model_ckpt_path=ckpt_path, tile_size=512, overlap=256, batch_size=8, device="cuda")
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
    g = cg
    xa, ya = geodetic_to_fixed_grid(cl[g], clo[g], sat_a, h_m=0.)
    cf, rf = scanning_angle_to_pixel(xa, ya, sat_a)
    ci, ri = np.round(cf).astype(int), np.round(rf).astype(int)
    ib = (ci >= 0) & (ci < sat_a.n_cols) & (ri >= 0) & (ri < sat_a.n_rows) & np.isfinite(cf)
    ci, ri = np.clip(ci, 0, sat_a.n_cols - 1), np.clip(ri, 0, sat_a.n_rows - 1)
    ok = ib & (qf[ri, ci] > 0) & np.isfinite(h[ri, ci]) & np.isfinite(u_ms[ri, ci]) & qa[ri, ci]
    n = ok.sum()
    m = dict(ch=ch[g][ok], cu=cu[g][ok], cv=cv[g][ok],
             ah=h[ri[ok], ci[ok]], au=u_ms[ri[ok], ci[ok]], av=v_ms[ri[ok], ci[ok]])
    hr = height_rmse(m["ah"], m["ch"])
    hb = float(np.mean(m["ah"] - m["ch"]))
    rv = rmsvd(m["au"], m["av"], m["cu"], m["cv"])
    sb = speed_bias(m["au"], m["av"], m["cu"], m["cv"])
    hc = correlation(m["ah"], m["ch"])
    print(f"{label:45s}  {n:>6,}  {hr:>7.0f}m  {hb:>+7.0f}m  {hc:>6.4f}  {rv:>6.2f}m/s  {sb:>+6.2f}m/s")


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

#!/usr/bin/env python
"""Sample ERA5 winds along the EarthCARE track for the curtain figure's ERA5
panel (scripts/fig_earthcare_curtain.py --era5-from ...).

Reads the along-track lat/lon/dist from a curtain bundle (or a track npz),
pulls one ERA5 analysis time (arco-era5 GCS via stereo_winds.validation.era5),
and for track points spaced ~--stride-km apart records the full ERA5 wind
column (every pressure level) placed at each level's geometric height.
Output npz: e_dp (km), e_hp (m), e_up, e_vp (m/s) — consumed by --era5-from.

    python scripts/sample_era5_curtain.py --bundle figures/curtain_bundle.npz \
        --time 2025-11-07T21:00 --out figures/era5_curtain.npz
"""
import argparse
import numpy as np

from stereo_winds.validation.era5 import open_era5_reader, load_era5_single_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", help="curtain bundle npz (uses its lat/lon/dist track)")
    ap.add_argument("--track", help="alt: npz with lat/lon/dist arrays")
    ap.add_argument("--time", required=True, help="ERA5 analysis time (ISO)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride-km", type=float, default=60.0)
    ap.add_argument("--hmax-km", type=float, default=18.0)
    args = ap.parse_args()

    src = np.load(args.bundle or args.track, allow_pickle=True)
    lat = src["lat"].astype(float); lon = src["lon"].astype(float)
    dist = src["dist"].astype(float)

    keep = [0]
    for i in range(1, dist.size):
        if dist[i] - dist[keep[-1]] >= args.stride_km:
            keep.append(i)
    keep = np.array(keep)
    print(f"track pts {dist.size} -> subsampled {keep.size}")

    reader = open_era5_reader()
    e = load_era5_single_time(reader, np.datetime64(args.time),
                              lat_bbox=(lat.min() - 1, lat.max() + 1),
                              lon_bbox=(lon.min() - 1.5, lon.max() + 1.5))
    u_all = e["u_component_of_wind"].values      # (level, lat, lon)
    v_all = e["v_component_of_wind"].values
    h_all = e["geometric_height"].values
    elat = e["lat"].values; elon = e["lon"].values

    dp, hp, up, vp = [], [], [], []
    hmax = args.hmax_km * 1000.0
    for i in keep:
        ai = int(np.argmin(np.abs(elat - lat[i])))
        oi = int(np.argmin(np.abs(elon - lon[i])))
        hc = h_all[:, ai, oi]; uc = u_all[:, ai, oi]; vc = v_all[:, ai, oi]
        for L in range(hc.size):
            if np.isfinite(hc[L]) and 0.0 <= hc[L] <= hmax and np.isfinite(uc[L]):
                dp.append(dist[i]); hp.append(hc[L]); up.append(uc[L]); vp.append(vc[L])
    dp, hp, up, vp = (np.asarray(a, np.float32) for a in (dp, hp, up, vp))
    np.savez(args.out, e_dp=dp, e_hp=hp, e_up=up, e_vp=vp, time=args.time)
    print(f"wrote {args.out}: {dp.size} barbs, h[m] {hp.min():.0f}-{hp.max():.0f}")


if __name__ == "__main__":
    main()

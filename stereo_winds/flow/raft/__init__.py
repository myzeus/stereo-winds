"""Single-channel RAFT optical-flow network.

Adapted from RAFT (Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms
for Optical Flow", ECCV 2020; https://github.com/princeton-vl/RAFT, BSD-3-Clause)
for single-channel geostationary imagery (``input_dim=1``). See NOTICE.
"""
from stereo_winds.flow.raft.raft import RAFT

__all__ = ["RAFT"]

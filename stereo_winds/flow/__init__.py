"""Self-contained RAFT optical-flow stage for stereo wind retrieval.

``FlowRunner`` runs a single-channel RAFT model over tiled geostationary
imagery to produce dense disparity fields. The RAFT network (``flow.raft``) is
vendored so stereo-winds has no external optical-flow dependency.
"""
from stereo_winds.flow.raft.raft import RAFT
from stereo_winds.flow.runner import FlowRunner, RAFT_ARGS, image_histogram_equalization

__all__ = ["FlowRunner", "RAFT", "RAFT_ARGS", "image_histogram_equalization"]

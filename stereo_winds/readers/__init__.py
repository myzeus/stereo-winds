"""Standalone satellite data readers for stereo-winds.

``GOES`` reads GOES-R ABI L1b radiance from NOAA's public S3 buckets without
authentication or satpy. Additional sensors (Himawari AHI, MTG-I FCI) are
optional and require their own extras.
"""
from stereo_winds.readers.goes import GOES

__all__ = ["GOES"]

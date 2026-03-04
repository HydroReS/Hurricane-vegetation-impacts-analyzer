"""
hurricane_vegetation_analysis.src
==================================
Core modules for satellite-based vegetation impact analysis.

Modules
-------
utils               — Configuration loading, ROI parsing, GEE initialization
data_acquisition    — Sentinel-2 / Landsat collection retrieval and compositing
vegetation_indices  — NDVI, EVI, SAVI, NDMI computation on ee.Image objects
analysis            — Difference maps, statistical tests, impact classification
visualization       — Interactive maps, histograms, HTML reports
metadata_utils      — Scene metadata extraction, warnings, and display helpers
"""

__version__ = "1.0.0"

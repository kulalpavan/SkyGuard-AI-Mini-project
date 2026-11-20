# Test fixtures — real district data, committed for CI

`udupi_weather.csv` and `dakshina_kannada_weather.csv` are the
district-mean hourly series (timestamp_epoch, t2m_deg_c,
relative_humidity_pct, msl_hpa) derived from the supplied district
files (NOAA source, shared 2026-11-19): one row per hourly timestamp
(2,617 rows each), values averaged over the district's grid points.

They are written at full float precision, so `load_district_series`
returns the same values as reading the raw per-grid files — the
backend Mamba pipeline tests use them for backfill and Mamba
registration exactly as they used the raw files.

Why they live here: the tests previously read the raw CSVs from a
`data/` folder next to the repo checkout, which only existed on one
machine. In a fresh checkout (CI) that folder does not exist and the
two Mamba pipeline tests failed with FileNotFoundError. Committing
this small derived fixture (~320 KB) makes the tests self-contained.

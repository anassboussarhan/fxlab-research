# Round 15 part D: probe the free sources (availability, format, date range only).
pip install -q requests pandas pyarrow numpy >/dev/null 2>&1
mkdir -p out/probe15b && python ci/probe15.py out/probe15
# the calendar archives are kept as artifacts only (licence: MIT, but stored outside the published branch)
mkdir -p hfkeep && mv out/probe15b/hf hfkeep/ 2>/dev/null || true

# Round 9 data collection (docs/round9_preregistration.md): raw files only, parsers come after the formats are seen.
pip install -q akshare lxml html5lib || true
timeout 4800 python ci/r9_data.py out/r9data
du -sh out/r9data/* || true

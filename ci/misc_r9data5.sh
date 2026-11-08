sudo apt-get -qq install -y poppler-utils >/dev/null 2>&1 || true
python ci/r9_data5.py out/r9data

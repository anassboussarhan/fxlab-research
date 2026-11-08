# Puts the round 9 raw data (branches misc-r9data2, -3, -4; later ones overwrite earlier files) into ./r9data.
mkdir -p r9data
for b in misc-r9data2 misc-r9data3 misc-r9data4; do
  rm -rf r9tmp && mkdir -p r9tmp
  git fetch -q --depth 1 origin $b && git --work-tree=r9tmp checkout FETCH_HEAD -- misc && git reset -q
  for f in $(cd r9tmp/misc/r9data && find . -type f); do s=$(stat -c %s r9tmp/misc/r9data/$f); if [ "$s" -gt 600 ] || [ ! -e r9data/$f ]; then mkdir -p r9data/$(dirname $f); cp r9tmp/misc/r9data/$f r9data/$f; fi; done
done
ls -R r9data | head -40

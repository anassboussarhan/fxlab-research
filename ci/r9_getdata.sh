# Puts the round 9 raw data (branch misc-r9data2) into ./r9data.
git fetch -q --depth 1 origin misc-r9data2 && mkdir -p r9tmp && git --work-tree=r9tmp checkout FETCH_HEAD -- misc && mv r9tmp/misc/r9data r9data && git reset -q && ls r9data

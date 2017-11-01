#!/bin/bash
awk  'NR % 2 == 1 {t=$0} NR % 2 == 0 {a[t]+=$(NF-1); c[t]+=1 } END{for (t in a) { print t, a[t]/c[t]/1000}}' $1 | sort -n

#!/bin/bash
ns="100 1000 10000 100000"
ds="10 100 1000 10000"
for N in $ns
do
  for d in $ds
  do
    run-scripts/gen-test-data-matrix-given.sh $d $N 9 --zip
  done
done


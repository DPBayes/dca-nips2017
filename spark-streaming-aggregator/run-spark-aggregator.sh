#!/bin/bash
res=$1
shift

rm -rf temp

class=fi.helsinki.cs.probic.streaming.SparkDataAggregator
./run-spark.sh $class \
  --input file://$PWD/../probic-decrypt-server/test-data-matrix \
  --output file://$PWD/temp/sum-data-matrix \
  --noise 9 $*
echo "9 $*" >> $res
tail -n 1 ${class}.log >> $res

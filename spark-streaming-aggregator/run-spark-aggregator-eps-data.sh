#!/bin/bash
res=results-eps-abalone-3000.txt
shift

rm -rf temp

fun(){
class=fi.helsinki.cs.probic.streaming.SparkDataAggregator
./run-spark.sh $class \
  --input file://$PWD/../dataset_tests/src/sparkfile.txt \
  --output file://$PWD/temp/eps-sum-data \
  --noise 9 $*
echo "9 $*" >> $res
tail -n 1 ${class}.log >> $res
}

fun --clients 3000 --d 8 --useDouble


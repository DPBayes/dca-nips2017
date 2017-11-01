#!/bin/bash

SPARK=$HOME/work/spark-2.1.0-bin-hadoop2.7

if [ -n "$1" ]
then
  class=$1
  shift
else
  echo "Usage: $0 classname [args...]"
  exit 1
fi

export SPARK_LOCAL_IP="127.0.0.1"
export SPARK_LOCAL_DIRS="/run/user/$( id -u $USER )/spark"
echo SPARK_LOCAL_DIRS=$SPARK_LOCAL_DIRS
echo "args: $class $*"

$SPARK/bin/spark-submit --driver-memory 1500g --master "local[45]" \
 --class $class $PWD/target/scala-2.11/probic-streaming-aggregator.jar $* 1> "${class}.log" 2> "${class}.err"


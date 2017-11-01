#!/bin/bash 

if [ -z "$3" ]; then echo "Usage: $0 d N noise"; exit 1; fi

d=$1
shift
N=$1
shift
noise=$1
shift

java  -Xmx1500g -cp target/scala-2.11/probic-server.jar \
fi.helsinki.cs.probic.data.GenerateTestDataMatrix \
--dimension $d \
--clients $N \
--noise $noise \
--output test-data-matrix-$d-$N-$noise.csv \
$*


#!/bin/bash

genargs() {
  certs=""
  masters=""  
  for i in $( seq 1 $1 )
  do
    let p=8080+$i
    let p-=1
    if [ -z "$certs" ]; then
      certs="probic-$i"
      masters="localhost:$p"
    else
      certs="$certs,probic-$i"
      masters="$masters,localhost:$p"
    fi
  done
}

if [ -n "$1" ]
then 
  genargs $1
  shift
else
  genargs 5
fi
 
java -cp target/scala-2.11/probic-server.jar \
fi.helsinki.cs.probic.test.TestDataServer \
--certs ${certs} \
--masters ${masters} \
$*


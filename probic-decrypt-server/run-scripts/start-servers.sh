#!/bin/bash

startserver(){
  echo "Starting probic-server: $*"
  #xterm -e "java -jar target/scala-2.11/probic-server.jar $*" &
  screen -d -m -S probic-server -- nice -n 20 java -Xmx10g -jar target/scala-2.11/probic-server.jar $* &
}

if [ -z "$1" ]; then c=5; else c=$1; fi
if [ -z "$2" ]; then msg=100; else msg=$2; fi

for i in $( seq 1 $c )
do
  let p=8080+$i
  let p--
  startserver --port $p --cert probic-$i --messages $msg
done


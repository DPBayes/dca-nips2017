#!/bin/bash

. ./gen-keys.sh

for i in $( seq 1 10 )
do
  genkey "$i"
done

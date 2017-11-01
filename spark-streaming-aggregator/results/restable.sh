#!/bin/bash
ns="100 1000 10000 100000"
ds="10 100 1000 10000"

mktable(){
echo '\begin{table}'
echo '\begin{tabular}[]{c c c c c}'
echo "M=$M & N=100 & N=1000 & N=10000 & N=100000 \\\\"
for d in $ds
do
  row="d=$d"
  for N in $ns
  do
    f=result-$d-$N-$M.txt
    if [ ! -f $f ]; then cell=NA; else
      cell=$( tail -n 1 $f | awk '{print $1}' )
    fi
    if [ -z "$row" ]; then row=$cell; else row="$row & $cell"; fi
  done
  echo "$row \\\\"
done
echo '\end{tabular}'
echo '\end{table}'
}

M=5
mktable
echo ""

M=10
mktable


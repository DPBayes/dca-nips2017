#!/bin/bash 
java -cp target/scala-2.11/probic-server.jar \
fi.helsinki.cs.probic.data.TestDataWriter \
--certs probic-1,probic-2,probic-3,probic-4,probic-5 \
--masters localhost:8080,localhost:8081,localhost:8082,localhost:8083,localhost:8084 \
--clients 10 \
--input test-data-matrix.csv \
--output test-data-matrix-crypt.csv

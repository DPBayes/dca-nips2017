# Introduction: Spark Streaming Aggregator Subprojects

The Probic Spark Streaming Data Aggregator consists of two projects, spark-streaming-aggregator that routes
data to the correct node for decryption, and probic-decrypt-server that represents one such decryption node.
The node then decrypts the data and returns the result to Spark which aggregates results from all such nodes
and produces the sum until input is exhausted (or forever, this can be adjusted).

## Requirements

Both projects use sbt as the build tool. Usage:

1. get sbt from http://www.scala-sbt.org/download.html , then extract it, and put its bin folder into your path.

2. `cd spark-streaming-aggregator` or `cd probic-decrypt-server`

3. `sbt eclipse` creates an Eclpse project file that allows you to import the spark-streaming-aggregator directory as a scala-ide project: http://scala-ide.org/

5. `sbt assembly` creates a so called fat jar that can be copied to any machine with Java and Spark installed and run as a Spark Streaming program.

6. Spark 2.1.0 or newer prebuilt for Hadoop 2.7 is needed by the tests. We assume Spark is downloaded from http://spark.apache.org/ , extracted, and placed at `$HOME/work/spark-2.1.0-bin-hadoop2.7` . If Spark is placed elsewhere, please adjust the file `spark-streaming-aggregator/run-spark.sh` accordingly.


# Running the experiment in the paper

After compiling, the sections below explains how to run the DCA experiment used to generate the results in Table 1 in the NIPS 2017 paper Differentially private Bayesian learning on distributed data.

The process includes some preparation and data generation steps followed by starting the decryption servers, and finally the top-level Spark-based data aggregator.

## Create keys if not yet done

This requires java's keytool to be installed.

1. Make a file called secret.txt with a one-line password that will be used for server keys. Currently the same password is used for all.
2. Run `./gen10.sh` to generate 10 public/private key pairs.

## Generate testing data file
Run `run-scripts/gen-testdata-10.sh` to generate a test data file for 10 decryption servers, 10:1 noise to real data message ratio, N=100 to 100,000 and d=10 to 10,000.

## Start the aggregators
Run `run-scripts/start-servers.sh n C` where n is the number of servers and equal to amount of noise, 10 above) and C is the number of clients. This will start the decryption servers. They will wait for the Spark process to act as the clients and schedule the data processing for them.

## Start the Spark aggregator
cd to `../spark-streaming-aggregator` and run:

```
sbt assembly
for d in 10; do for k in $( seq 1 5 ); do ./run-spark-aggregator4.sh results-agg4.txt --d $d --clients 100 --repeats 10; done ; done
```
Note: To run the whole experiment, you need to then kill the aggregator server processes, and restart with --clients 1000, then 10,0000, etc. until the whole table of results has been generated.

# Spark Non-Streaming Aggregator Final Results

You can obtain these 5-run averages using `./getresults.sh results-agg4.txt` in the `spark-streaming-aggregator` folder.
The output should look like this:
```
9 --d 10000 --clients 10000 1103.47
9 --d 10000 --clients 1000 93.5316
9 --d 10000 --clients 100 19.8942
9 --d 1000 --clients 100000 666.504
9 --d 1000 --clients 10000 109.383
9 --d 1000 --clients 1000 11.7662
9 --d 1000 --clients 100 3.8872
9 --d 100 --clients 100000 70.3782
9 --d 100 --clients 10000 12.7582
9 --d 100 --clients 1000 3.021
9 --d 100 --clients 100 2.1504
9 --d 10 --clients 100000 8.71983
9 --d 10 --clients 10000 3.0218
9 --d 10 --clients 1000 2.129
9 --d 10 --clients 100 1.8676
```

# Running an experiment with a real dataset based model

## Generate sufficient statistics + noise
```sh
cd ../dataset_tests/src
python3 eps_data_test.py -s sparkfile.txt-8-3000-9.csv -c 10
```
And compress it for Spark:
```
gzip sparkfile.txt-8-3000-9.csv
```

## Run decryption servers
First, compile the project with `sbt assembly`. Then run:

```sh
cd ../../probic-decrypt-server
run-scripts/start-servers-eps.sh
```

## Run Spark aggregator
First, compile the project with `sbt assembly`. Then run:

```
cd ../spark-streaming-aggregator
./run-spark-aggregator-eps-data.sh
```

Results will be produced in the spark-streaming-aggregator folder in the file results-eps-abalone-3000.txt.
Run as many repeats as you wish.
You can get the average runtime by running
```sh
./get-results.sh results-eps-abalone-3000.txt 
```
The results may be something like:
```sh
9 --clients 3000 --d 8 --useDouble 6.3494
```
In this case the experiment took 6.35 seconds to complete on average.

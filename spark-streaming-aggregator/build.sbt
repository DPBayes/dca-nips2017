name := "Probic Spark Streaming Private Data Aggregator"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.1" % "provided"

libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.1.1" % "provided"

libraryDependencies += "fi.helsinki.cs.nodes" % "getopt-scala" % "1.1.0"

libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0"

// https://mvnrepository.com/artifact/commons-codec/commons-codec
libraryDependencies += "commons-codec" % "commons-codec" % "1.10"

libraryDependencies += "org.slf4j" % "slf4j-simple" % "1.7.25"

assemblyJarName in assembly := "probic-streaming-aggregator.jar"

mainClass in assembly := Some("fi.helsinki.cs.probic.streaming.Aggregator")


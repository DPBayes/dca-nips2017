name := "Probic Private Data Aggregation Node"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies += "fi.helsinki.cs.nodes" % "getopt-scala" % "1.1.0"

libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.4.0"

// https://mvnrepository.com/artifact/commons-codec/commons-codec
libraryDependencies += "commons-codec" % "commons-codec" % "1.10"

libraryDependencies += "org.slf4j" % "slf4j-simple" % "1.7.25"

mainClass in assembly := Some("fi.helsinki.cs.probic.server.Server")

assemblyJarName in assembly := "probic-server.jar"

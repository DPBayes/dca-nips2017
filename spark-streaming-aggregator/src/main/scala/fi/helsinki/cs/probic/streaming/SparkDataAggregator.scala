package fi.helsinki.cs.probic.streaming

import fi.helsinki.cs.nodes.util.Spark2Main
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.dstream.DStream
import com.typesafe.scalalogging.LazyLogging
import java.net.URL
import java.net.Socket
import java.io.DataOutputStream
import org.apache.commons.codec.binary.Base64
import java.io.DataInputStream
import scala.io.Codec
import org.apache.commons.io.FileUtils
import java.io.File

/**
 * Assumes data is already in correct order, e.g. data for first server first, then the data for 2nd server, etc.
 *
 * Mandatory options:
 * --intype hdfs or --intype socket
 * --input hdfs://path/to/input/folder
 * --output hdfs://path/to/output/foldersrc/main/scala/fi/helsinki/cs/probic/
 */
object SparkDataAggregator extends Spark2Main with LazyLogging {
  val longOptions = Seq("clients=", "input=", "output=", "noise=", "d=", "repeats=", "useDouble")

  val shortOptions = ""

  val sparkOutputCompression = true

  def sparkMain(spark: SparkSession) {
    // test-data-matrix
    val input = mandatoryOption("input")
    val output = mandatoryOption("output")
    val d = mandatoryOption("d").toInt
    val clients = mandatoryOption("clients").toInt
    val noise = mandatoryOption("noise").toInt
    val useDouble = optionSet("useDouble")
    val k = noise + 1

    var timeAcc = 0L
    val repeats = optional("repeats").getOrElse("1").toInt
    val out = s"$output-$d-$clients-$noise.csv.gz"
    import sys.process._
    val result = "rm -rf temp" !
    val start = System.currentTimeMillis
    val in = spark.sparkContext.textFile(s"$input-$d-$clients-$noise.csv.gz", d).repartition(d).zipWithIndex

    val resultStream = {
      if (useDouble) {
        // Matrix is N lines, each line has (noise+1)*D messages
        // So we need to gather each group of (noise+1) items on each line to produce full batches of messages.
       val dGroupedLines = in.flatMap {
          case (line, clientId) =>
            val dGroups = line.split(";").map(_.toDouble).grouped(k).toSeq.zipWithIndex
            dGroups.map{case (kItems, dValue) =>
              dValue -> kItems.zipWithIndex.map{case (item, serverId) => (serverId, clientId, item)}
            }
        }
       dGroupedLines.reduceByKey(_++_).flatMap{case (dValue, batch) =>
         val byServer = batch.toSeq.groupBy(_._1).map{x => x._2.map(_._3).toArray -> x._1}
         val outputs = (0 until repeats).toSeq.par.map{ repeatId =>
             val output = byServer.toSeq.par.map(sendReceive(useDouble)).reduce(_+_)
             val adjustedLineId = (dValue * 10 + repeatId)
              s"$adjustedLineId;$output"
       }
         outputs.seq
       }
      } else {
        // Matrix is D lines, each line has (noise+1)*N messages
        in.flatMap {
          case (line, lineNum) =>
            // One line is a whole batch of messages, send to servers for sum.
            val outputs = (0 until repeats).toSeq.par.map { repeatId =>
              val output = line.split(";").map { x =>
                if (useDouble)
                  x.toDouble
                else
                  x.toLong
              }.grouped(clients).toSeq.zipWithIndex.par.map(sendReceive(useDouble))
                .reduce(_ + _)
              val adjustedLineId = (lineNum * 10 + repeatId)
              s"$adjustedLineId;$output"
            }
            outputs.seq
          //lineNum -> output
        }
      }
    }

    //resultStream.map { case (k, value) => k + ";" + value }
    resultStream.saveAsTextFile(out)
    val end = System.currentTimeMillis()
    timeAcc += end - start

    println(s"Total time: $timeAcc ms.")
  }

  def sendReceive(useDouble: Boolean)(valuesForServer: (Array[Double], Int)) = {
    val (values, srvId) = valuesForServer
    val sock = new Socket("127.0.0.1", 8080 + srvId)
    val out = new DataOutputStream(sock.getOutputStream)
    if (useDouble)
      values.foreach(out.writeDouble)
    else
      values.map(_.toLong).foreach(out.writeLong)
    val in = new DataInputStream(sock.getInputStream)

    val returned = {
      if (useDouble)
        in.readDouble
      else
        in.readLong
    }
    sock.close
    returned
  }
}

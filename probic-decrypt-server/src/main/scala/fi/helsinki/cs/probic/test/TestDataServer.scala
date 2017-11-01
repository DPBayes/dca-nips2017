package fi.helsinki.cs.probic.test

import fi.helsinki.cs.nodes.util.OptMain
import com.typesafe.scalalogging.LazyLogging
import java.net.ServerSocket
import java.net.Socket
import org.apache.commons.codec.binary.Base64
import java.io.DataOutputStream
import fi.helsinki.cs.probic.crypto.PkCrypto
import java.util.zip.GZIPInputStream
import java.io.FileInputStream
import java.io.File
import scala.collection.Seq

/**
 * Mandatory options:
 * --intype hdfs or --intype socket
 * --input hdfs://path/to/input/folder
 * --output hdfs://path/to/output/foldersrc/main/scala/fi/helsinki/cs/probic/
 */
object TestDataServer extends OptMain with LazyLogging {

  val DEFAULT_PORT = "8090"

  val longOptions = Seq("port=", "masters=", "certs=", "input=", "clients=", "zip", "batchLength=", "sleep=")

  val shortOptions = ""

  def optMain() {
    val port = optional("port").getOrElse(DEFAULT_PORT).toInt
    val certs = optional("certs").getOrElse("probic").split(",")
    val masters = optional("masters").getOrElse("localhost:8080").split(",")
    val clients = mandatoryOption("clients").toInt
    val batchLength = optional("batchLength").getOrElse("5000").toLong
    val sleep = optional("sleep").getOrElse("500").toLong

    val input = mandatoryOption("input")
    val zip = optionSet("zip")

    val server = new ServerSocket(port)
    val crypto = new PkCrypto("probic") // Private test key, not relevant in this program

    val servers = for (i <- 0 until certs.length) yield {
      masters(i) -> crypto.getEncrypter(certs(i))
    }

    val handler = handleRequest(servers) _
    //val handler = handleRequestLine(servers) _

    val inputLines = {
      if (zip) {
        io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(new File(input)))).getLines()
      } else
        io.Source.fromFile(input).getLines()
    }

    logger.info(s"Starting ${getClass.getName} at $port")
    handler(server.accept, inputLines, clients, batchLength, sleep)
    /*for (line <- inputLines)
      handler(server.accept, line, clients)*/
  }

  def handleRequest(servers: Seq[(String, String => Array[Byte])])(sock: Socket, inputLines: Iterator[String], clients: Int, batchLength: Long, sleep: Long) {
    val t0 = System.currentTimeMillis()
    var batchId = 0
    val out = new DataOutputStream(sock.getOutputStream)
    for (line <- inputLines) {
      timedWriteout(out, servers, line, clients)
      var diff = System.currentTimeMillis() - t0
      logger.info(s"Total elapsed ${diff} ms.")
      diff -= batchId * batchLength
      val slp = batchLength - diff + sleep // 500 to make sure
      logger.info(s"Sleeping $slp ms to compensate")
      Thread.sleep(slp)
      batchId += 1
    }
    out.close
  }

  def handleRequestLine(servers: Seq[(String, String => Array[Byte])])(sock: Socket, line: String, clients: Int) {
    val out = new DataOutputStream(sock.getOutputStream)
    parWriteout(out, servers, line, clients)
    out.close
  }

  def timedWriteout(out: DataOutputStream, servers: Seq[(String, String => Array[Byte])], line: String, clients: Int) = {
    val t1 = System.currentTimeMillis()
    val items = line.split(";")
    val itemsPerClient = items.length / clients
    val clientItems = items.grouped(itemsPerClient).toSeq
    for (client <- 0 until clients) {
      val myItems = clientItems(client)
      for (item <- 0 until myItems.length) {
        val (master, encrypt) = servers(item % servers.length)
        val data = myItems(item)
        val cryptoText = encrypt(data + "")
        val msg = s"$master;$client;${new String(Base64.encodeBase64(cryptoText))}"
        out.write((msg + "\n").getBytes)
      }
    }
    logger.info(s"Sent data of $clients clients with $itemsPerClient items per client in ${System.currentTimeMillis() - t1} ms.")
  }

  def parWriteout(out: DataOutputStream, servers: Seq[(String, String => Array[Byte])], line: String, clients: Int) = {
    val t1 = System.currentTimeMillis()
    val items = line.split(";")
    val itemsPerClient = items.length / clients
    val groupsOfServers = items.grouped(itemsPerClient).zipWithIndex.toSeq
    val encrypted = servers.zipWithIndex.par.flatMap {
      case ((master, encrypt), sindex) =>
        groupsOfServers.flatMap {
          case (group, client) =>
            //println(s"sindex $sindex grouplen ${group.length} client $client")
            val data = group(sindex)
            val cryptoText = encrypt(data + "")
            val msg = s"$master;$client;${new String(Base64.encodeBase64(cryptoText))}\n"
            msg.getBytes
        }
    }
    encrypted.seq.foreach { msg =>
      out.write(msg)
    }
    logger.info(s"Sent data of $clients clients with $itemsPerClient items per client in ${System.currentTimeMillis() - t1} ms.")
  }
}

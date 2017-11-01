package fi.helsinki.cs.probic.test

import fi.helsinki.cs.nodes.util.OptMain
import com.typesafe.scalalogging.LazyLogging
import org.apache.commons.codec.binary.Base64
import fi.helsinki.cs.probic.crypto.PkCrypto
import fi.helsinki.cs.probic.data.GenerateTestDataMatrix
import scala.collection.Seq

/**
 * Mandatory options:
 * --intype hdfs or --intype socket
 * --input hdfs://path/to/input/folder
 * --output hdfs://path/to/output/foldersrc/main/scala/fi/helsinki/cs/probic/
 */
object TestDataWriter extends OptMain with LazyLogging {

  val longOptions = Seq("port=", "masters=", "certs=", "input=", "clients=", "output=")

  val shortOptions = ""

  def optMain() {
    val output = mandatoryOption("output")
    val certs = optional("certs").getOrElse("probic").split(",")
    val masters = optional("masters").getOrElse("localhost:8080").split(",")
    val clients = mandatoryOption("clients").toInt

    val input = mandatoryOption("input")

    val crypto = new PkCrypto("probic") // Private test key, not relevant in this program

    val servers = for (i <- 0 until certs.length) yield {
      masters(i) -> crypto.getEncrypter(certs(i))
    }
    val inputLines = io.Source.fromFile(input).getLines().toSeq
    val outputFile = inputLines.flatMap(line => getOutputLines(servers, line, clients).flatten)
    GenerateTestDataMatrix.toFile(output, outputFile)
  }

  def getOutputLines(servers: Seq[(String, String => Array[Byte])], line: String, clients: Int) = {
    val items = line.split(";")
    val itemsPerClient = items.length / clients
    val clientItems = items.grouped(itemsPerClient).toSeq
    for (client <- 0 until clients) yield {
      val myItems = clientItems(client)
      for (item <- 0 until myItems.length) yield {
        val (master, encrypt) = servers(item % servers.length)
        val data = myItems(item)
        val cryptoText = encrypt(data + "")
        val msg = s"$master;$client;${new String(Base64.encodeBase64(cryptoText))}"
        msg
      }
    }
  }
}

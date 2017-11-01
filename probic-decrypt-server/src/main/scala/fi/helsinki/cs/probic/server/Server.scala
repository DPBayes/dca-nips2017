package fi.helsinki.cs.probic.server

import fi.helsinki.cs.nodes.util.OptMain
import com.typesafe.scalalogging.LazyLogging
import java.net.ServerSocket
import java.net.Socket
import scala.concurrent.ExecutionContext
import scala.concurrent.Future
import java.io.ByteArrayInputStream
import java.io.InputStreamReader
import java.io.ByteArrayOutputStream
import java.security.KeyStore
import javax.crypto.Cipher
import java.security.spec.X509EncodedKeySpec
import java.security.KeyFactory
import java.security.PublicKey
import org.apache.commons.codec.binary.Base64
import java.io.DataOutputStream
import java.io.DataInputStream
import fi.helsinki.cs.probic.crypto.PkCrypto
import scala.concurrent.forkjoin.ForkJoinPool
import sun.misc.VM
import scala.collection.parallel.ForkJoinTaskSupport

/**
 * Mandatory options:
 * --intype hdfs or --intype socket
 * --input hdfs://path/to/input/folder
 * --output hdfs://path/to/output/foldersrc/main/scala/fi/helsinki/cs/probic/
 */
object Server extends OptMain with LazyLogging {

  val DEFAULT_PORT = "8080"

  val longOptions = Seq("port=", "cert=", "messages=", "useDouble")

  val shortOptions = ""

  def optMain() {

    val port = optional("port").getOrElse(DEFAULT_PORT).toInt
    val cert = optional("cert").getOrElse("probic")
    // How many messages to receive before decrypting and returning a result.
    val messages = mandatoryOption("messages").toInt

    val useDouble = optionSet("useDouble")

    val server = new ServerSocket(port)
    //val pk = new PkCrypto(cert)

    val handler = handleRequestStreaming(messages, useDouble) _
    logger.info(s"Starting Probic Data Aggregation Server at $port")
    //logger.info("Available processors: " + Runtime.getRuntime.availableProcessors() + ", using only 5")
    var running = true
    while (running) {
      handler(server.accept)
    }
  }

  var decryptedMessages = 0
  var decryptedSum = 0.0

  def handleRequest(messages: Int)(sock: Socket) {
    implicit val ec = ExecutionContext.global
    val answer = Future {
      val src = new DataInputStream(sock.getInputStream)
      // Read and decrypt a total of `messages` messages.

      // Sequentially read messages:
      val msgSeq = (0 until messages).map { msgId =>
        /*val len = src.readInt()
        val cryptoText = new Array[Byte](len)
        src.read(cryptoText)
        //logger.info(s"Received msg id $msgId")
        cryptoText*/
        src.readLong
      }.toSeq

      // Decrypt them in parallel using 5 threads
      /*val outValue = msgSeq.grouped(messages / 10).toSeq.par.flatMap { group =>
        /*val rsa = pk.getDecrypt()
        group.map {
          cryptoText =>
            // This is thread safe
            val msg = new String(rsa.doFinal(cryptoText))
            msg.toDouble
        }*/
      }.reduce(_ + _)*/

      val outValue = msgSeq.par.reduce(_ + _)

      logger.info(s"Decrypted $messages messages, returning $outValue")
      val out = new DataOutputStream(sock.getOutputStream)
      out.writeLong(outValue)
      sock.close()
      outValue
    }
  }

  def handleRequestStreaming(messages: Int, useDouble: Boolean)(sock: Socket) {
    implicit val ec = ExecutionContext.global
    val answer = Future {
      val src = new DataInputStream(sock.getInputStream)
      // Read and decrypt a total of `messages` messages.
      val outValue = {
        if (useDouble) {
          var result = 0.0
          // Sequentially read messages:
          val msgSeq = (0 until messages).foreach { msgId =>
            result += src.readDouble
          }
          result
        } else {
          var result = 0L
          // Sequentially read messages:
          val msgSeq = (0 until messages).foreach { msgId =>
            result += src.readLong
          }
          result
        }
      }

      logger.info(s"Decrypted $messages messages, returning $outValue")
      val out = new DataOutputStream(sock.getOutputStream)
      if (useDouble)
        out.writeDouble(outValue)
      else // Possible loss of precision.
        out.writeLong(outValue.toLong)
      sock.close()
      outValue
    }
  }

  def handleRequestStreamingDouble(messages: Int)(sock: Socket) = {
    implicit val ec = ExecutionContext.global
    Future {
      val src = new DataInputStream(sock.getInputStream)
      // Read and decrypt a total of `messages` messages.
      var result = 0.0
      // Sequentially read messages:
      val msgSeq = (0 until messages).foreach { msgId =>
        result += src.readDouble
      }
      val outValue = result

      logger.info(s"Decrypted $messages messages, returning $outValue")
      val out = new DataOutputStream(sock.getOutputStream)
      out.writeDouble(outValue)
      sock.close()
      outValue
    }
  }
}

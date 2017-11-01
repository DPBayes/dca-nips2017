package fi.helsinki.cs.probic.test

import fi.helsinki.cs.nodes.util.OptMain
import com.typesafe.scalalogging.LazyLogging
import java.net.Socket
import java.io.DataInputStream
import java.io.DataOutputStream
import fi.helsinki.cs.probic.crypto.PkCrypto
import scala.collection.Seq

/**
 * Test the server by sending it encrypted messages forever.
 */
object TestClient extends OptMain with LazyLogging {

  val DEFAULT_PORT = "8080"

  val longOptions = Seq("port=")

  val shortOptions = ""

  def optMain() {
    val port = optional("port").getOrElse(DEFAULT_PORT).toInt

    val crypto = new PkCrypto("probic")
    val encrypt = crypto.getEncrypter("probic")
    for (i <- 0 until 1000) {
      val plainText = s"Test Number $i"
      logger.info(plainText)
      val cryptoText = encrypt(plainText)
      val sock = new Socket("localhost", port)
      val out = new DataOutputStream(sock.getOutputStream)
      out.writeInt(cryptoText.length)
      out.write(cryptoText)
      val in = new DataInputStream(sock.getInputStream)
      val returned = in.readUTF()
      sock.close
      logger.info(s"Server returned: $returned")
      assert(plainText == returned)
    }
  }
}

package fi.helsinki.cs.probic.data

import fi.helsinki.cs.nodes.util.OptMain
import com.typesafe.scalalogging.LazyLogging
import java.net.ServerSocket
import java.net.Socket
import scala.concurrent.ExecutionContext
import org.apache.commons.codec.binary.Base64
import java.io.DataOutputStream
import fi.helsinki.cs.probic.crypto.PkCrypto
import java.util.zip.GZIPOutputStream
import scala.util.Random

/**
 * Mandatory options:
 * --intype hdfs or --intype socket
 * --input hdfs://path/to/input/folder
 * --output hdfs://path/to/output/foldersrc/main/scala/fi/helsinki/cs/probic/
 */
object GenerateTestDataMatrix extends OptMain with LazyLogging {

  val longOptions = Seq("dimension=", "clients=", "noise=", "output=", "zip")

  val shortOptions = ""

  def optMain() {
    val clients = mandatoryOption("clients").toInt
    val d = mandatoryOption("dimension").toInt
    val noise = mandatoryOption("noise").toInt
    val zip = optionSet("zip")

    val output = mandatoryOption("output")

    logger.info(s"Generating test data matrix of size d=$d x N=$clients x k=${noise + 1}")
    //generateAllData(output, d, clients, noise)
    generateAllDataLive(output, d, clients, noise, zip)
  }

  def generateAllData(output: String, d: Int, clients: Int, noise: Int) = {
    val (realData, confusedData) = generateData(d, clients, noise)
    toFile(output, confusedData.seq.map(_.mkString(";")))
    // Save also real data.
    toFile(s"$output-realdata.csv", realData.seq.map(_.mkString(";")))
    // Save sums for checking.
    toFile(s"$output-sums.csv", confusedData.seq.map(_.sum.toString))
    // Save sums for checking.
    toFile(s"$output-realdata-sums.csv", realData.seq.map(_.sum.toString))
  }

  def generateAllDataLive(output: String, d: Int, clients: Int, noise: Int, zip: Boolean) {
    // test data writer
    val writer = fileWriter(output, zip)
    // Realdata Sums writer
    val realWriter = fileWriter(s"$output-realdata-sums.csv", zip)
    val allData = generateDataLive(d, clients, noise)
    allData.map { line =>
      val (realData, confusedData) = line.unzip
      writer.write(confusedData.flatten.mkString(";") + "\n")
      realWriter.write(realData.sum + "\n")
    }.force
    writer.close
    realWriter.close
  }

  def generateData(d: Int, clients: Int, noise: Int) = {
    val lines = 0 until d
    val outputs = lines.par.map { l =>
      val allData = (0 until clients).map { client =>
        clientData(client, noise)
      }
      val (realData, confusedData) = allData.unzip
      val wholeLine = confusedData.flatten

      /*val rs = realData.sum
      val ws = wholeLine.sum
      assert(doubleEquals(rs, ws), s"$rs did not equal $ws. The sum of real data of the line should equal the sum of the confused data.")*/
      realData -> wholeLine
    }
    outputs.unzip
  }

  def generateDataLive(d: Int, clients: Int, noise: Int) = {
    val lines = 0 until d
    val outputs = lines.view.map { l =>
      val allData = (0 until clients).par.map { client =>
        clientData(client, noise)
      }
      allData
    }
    outputs
  }

  def clientData(clientId: Int, noise: Int) = {
    val rnd = new Random()
    def rlong() = {
      val lon = rnd.nextInt().toLong << 32
      lon + rnd.nextInt()
    }
    val plainText = rlong
    val noises = Seq.fill(noise)(rlong)

    val confusedRealData = plainText + noises.sum

    val clientData = (0 until noise + 1).view.map { j =>
      if (j == 0) { // "real" data
        confusedRealData
      } else
        noises(j - 1)
    }

    //assert(doubleEquals(plainText, clientData.sum), s"$plainText did not equal ${clientData.sum} for client $client. Real data should equal the sum of the confused data for each data item.")
    plainText -> clientData
  }

  /**
   * Store `lines` as a series of lines in a local file called `fileName`.
   */
  def toFile(fileName: String, lines: Iterable[String]) {
    toFile(fileName, lines, false)
  }

  /**
   * Store `lines` as a series of lines in a local file called `fileName`.
   */
  def toFile(fileNameBase: String, lines: Iterable[String], zip: Boolean = false) {
    val pw = fileWriter(fileNameBase, zip)

    lines.foreach(line => { pw.write(line + "\n") })
    pw.close()
  }

  /**
   * Store `lines` as a series of lines in a local file called `fileName`.
   */
  def fileWriter(fileNameBase: String, zip: Boolean) = {
    import java.io._
    val fileName = {
      if (zip) {
        s"${fileNameBase}.gz"
      } else
        s"${fileNameBase}"
    }

    val f = new File(fileName)
    val pw = {
      if (zip)
        new PrintWriter(new GZIPOutputStream(new FileOutputStream(f, false)))
      else
        new PrintWriter(f)
    }
    pw
  }
}

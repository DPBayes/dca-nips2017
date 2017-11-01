package fi.helsinki.cs.nodes.util

import org.apache.spark.sql.SparkSession

/**
 *
 * @author Eemil Lagerspetz
 */
trait Spark2Main extends OptMain {
  /**
   * Whether to compress Spark outputs. Required.
   */
  val sparkOutputCompression: Boolean

  /**
   * Main entry point. Configures Spark and parses args for options specified in `shortOptSpec` and `longOptSpec` (see getopt-scala docs).
   */
  def sparkMain(spark: SparkSession)

  /**
   * Main entry point. Configures Spark and parses args, then passes control to [[fi.helsinki.cs.nodes.carat.util.SparkMain#sparkMain]] .
   */
  def optMain() {
    val sb = SparkSession
      .builder()
      .appName(getClass.getName.replaceAll("$", ""))

    val spark = {
      if (sparkOutputCompression)
        enableCompression(sb).getOrCreate()
      else
        sb.getOrCreate()
    }

    sparkMain(spark)
  }

  private def enableCompression(sb: SparkSession.Builder) = {
    sb.config("spark.hadoop.mapred.output.compress", true)
      .config("spark.hadoop.mapred.output.compression.codec", true)
      .config("spark.hadoop.mapred.output.compression.codec", "org.apache.hadoop.io.compress.GzipCodec")
      .config("spark.hadoop.mapred.output.compression.type", "BLOCK")
  }
}

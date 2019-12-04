/* Wemakeprice Recommendation Project.
 *
 * Authors:
 * - Hyunsik Jeon (jeon185@snu.ac.kr)
 * - Jaemin Yoo (jaeminyoo@snu.ac.kr)
 * - U Kang (ukang@snu.ac.kr)
 * - Data Mining Lab. at Seoul National University.
 *
 * File: wemakeprice/io/package.scala
 * - Package object containing input and output methods.
 *
 * Version: 1.0.0
 */
package wemakeprice

import org.apache.spark.sql._
import org.apache.spark.sql.types._
import java.nio.file.Paths
import wemakeprice.columns._

package object io {
  // Schema used in reading useraction data.
  val useractionSchema = StructType(Seq(
    StructField("_c0", IntegerType, nullable = true),
    StructField("_c1", StringType, nullable = true),
    StructField("_c2", IntegerType, nullable = true),
    StructField("_c3", StringType, nullable = true),
    StructField("_c4", StringType, nullable = true)
  ))

  /**
    * Reads useraction data as a Spark DataFrame.
    *
    * @param dir Directory where useraction data lie. Every csv file in the directory will be read.
    * @return A Spark DataFrame containing useraction data.
    */
  def readUseraction(dir: String): DataFrame = {
    val df = spark.read
      .option("quote", "\"")
      .option("escape", "\"")
      .schema(useractionSchema)
      .csv(Paths.get(dir, "*.csv").toString)

    df.toDF(UseractionColumns: _*)
  }

  /**
    * Reads metadata as a Spark DataFrame.
    *
    * @param path File path of the metadata to read.
    * @return A Spark DataFrame containing metadata.
    */
  def readMetadata(path: String): DataFrame = {
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)
  }

  /**
    * Writes a Spark DataFrame as csv files.
    *
    * @param df     A Spark DataFrame to write.
    * @param dir    Directory at which csv files will be written.
    * @param header Whether to write header to file.
    */
  def write(df: DataFrame, dir: String, header: String): Unit = {
    df.write
      .option("quote", "\"")
      .option("escape", "\"")
      .option("header", header)
      .mode("overwrite")
      .csv(dir)
  }
}

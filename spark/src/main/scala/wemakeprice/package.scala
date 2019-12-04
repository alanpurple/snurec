/* Wemakeprice Recommendation Project.
 *
 * Authors:
 * - Hyunsik Jeon (jeon185@snu.ac.kr)
 * - Jaemin Yoo (jaeminyoo@snu.ac.kr)
 * - U Kang (ukang@snu.ac.kr)
 * - Data Mining Lab. at Seoul National University.
 *
 * File: wemakeprice/package.scala
 * - Package object containing Spark session object.
 *
 * Version: 1.0.0
 */
import org.apache.spark.sql._

package object wemakeprice {

  val spark: SparkSession = SparkSession
    .builder
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.eventLog.enabled", "true")
    .getOrCreate()
}

/* Wemakeprice Recommendation Project.
 *
 * Authors:
 * - Hyunsik Jeon (jeon185@snu.ac.kr)
 * - Jaemin Yoo (jaeminyoo@snu.ac.kr)
 * - U Kang (ukang@snu.ac.kr)
 * - Data Mining Lab. at Seoul National University.
 *
 * File: wemakeprice/feature/FeatureGenerator.scala
 * - A main class for generating feature data.
 *
 * Version: 1.0.0
 */
package wemakeprice.feature

import java.nio.file.Paths
import wemakeprice.io._
import wemakeprice.columns._
import wemakeprice.feature.functions._


object FeatureGenerator {
  def main(args: Array[String]) {
    /**
      * The main method gets total 7 arguments.
      * The arguments must be given in correct order.
      * 0: Directory where useraction data csv files lie.
      * 1: Directory where metadata csv files lie.
      * 2: Action type to create as feature data.
      * 3: Product frequency threshold. Any product not reaching the threshold will be filtered.
      * 4: Start date of feature data. Must be in YYYY-MM-DD format.
      * 5: End date of feature data.
      * 6: Directory where created feature data will be written.
      */
    val useractionDir = args(0)
    val metadataDir = args(1)
    val actionType = getActionType(args(2))
    val prodThreshold = args(3).toInt
    val (startTimestamp, endTimestamp) = getTimestampsFromString(args(4), args(5))
    val outputDir = args(6)

    // read data as a Spark DataFrame.
    val useractionDF = readUseraction(useractionDir)
    val userDF = readMetadata(Paths.get(metadataDir, "user.csv").toString)
    val productDF = readMetadata(Paths.get(metadataDir, "product.csv").toString)
    val prodCateDF = readMetadata(Paths.get(metadataDir, "prod_category.csv").toString)
    val categoryDF = readMetadata(Paths.get(metadataDir, "category.csv").toString)

    /**
      * Actual feature-creating method.
      *
      * @return A Spark DataFrame containing feature data.
      */
    def createFeature() = {
      val productDF2 = productDF
        .transform(join(prodCateDF, ProductNo))
        .transform(join(categoryDF, CategoryNo))

      useractionDF
        .transform(filterNullMemberId)
        .transform(filterByActionType(actionType))
        .transform(filterByTimestamp(startTimestamp, endTimestamp))
        .transform(upperMemberId)
        .transform(join(userDF, MemberId))
        .transform(join(productDF2, ProductNo))
        .transform(filterColumns)
        .transform(filterByProdThreshold(prodThreshold))
        .transform(sort(Timestamp))
    }

    // create feature data.
    val result = createFeature()

    // write feature data as csv files.
    write(result, outputDir, "true")
  }
}

/* Wemakeprice Recommendation Project.
 *
 * Authors:
 * - Hyunsik Jeon (jeon185@snu.ac.kr)
 * - Jaemin Yoo (jaeminyoo@snu.ac.kr)
 * - U Kang (ukang@snu.ac.kr)
 * - Data Mining Lab. at Seoul National University.
 *
 * File: wemakeprice/sort/Sorter.scala
 * - A main class for sorting useraction data.
 *
 * Version: 1.0.0
 */
package wemakeprice.sort

import wemakeprice.io._
import wemakeprice.feature.functions._

object Sorter {
  def main(args: Array[String]) {
    /**
      * The main method gets total 3 arguments.
      * The arguments must be given in correct order.
      * 0: Directory where useraction data csv files lie.
      * 1: Name of a column to sort with.
      * 2: Directory where sorted useraction data will be written.
      */
    val useractionDir = args(0)
    val sortColumn = args(1)
    val outputDir = args(2)

    // read data as a Spark DataFrame.
    val useractionDF = readUseraction(useractionDir)

    // sort data.
    val result = useractionDF.transform(sort(sortColumn))

    // write data as csv files.
    write(result, outputDir, "false")
  }
}

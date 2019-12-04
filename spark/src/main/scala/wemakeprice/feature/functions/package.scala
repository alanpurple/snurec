/* Wemakeprice Recommendation Project.
 *
 * Authors:
 * - Hyunsik Jeon (jeon185@snu.ac.kr)
 * - Jaemin Yoo (jaeminyoo@snu.ac.kr)
 * - U Kang (ukang@snu.ac.kr)
 * - Data Mining Lab. at Seoul National University.
 *
 * File: wemakeprice/feature/functions/package.scala
 * - Package object containing various functions for creating features.
 *
 * Version: 1.0.0
 */
package wemakeprice.feature

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import java.time.LocalDate
import wemakeprice.columns._

package object functions {
  /**
    * Get action type string that FeatureGenerator understands from user input.
    *
    * @param raw A raw input string of action type.
    * @return String indicating action type.
    */
  def getActionType(raw: String): String = {
    if (Seq("order", "o").contains(raw.toLowerCase)) "O"
    else if (Seq("click", "c").contains(raw.toLowerCase)) "C"
    else if (Seq("all", "a").contains(raw.toLowerCase)) "A"
    else throw new IllegalArgumentException(raw + " is not a proper action type!")
  }

  /**
    * Get POSIX timestamps from given strings.
    *
    * @param str1 A string indicating start date. Format must be YYYY-MM-DD.
    * @param str2 A string indication end date.
    * @return A tuple of timestamps.
    */
  def getTimestampsFromString(str1: String, str2: String): (Long, Long) = {
    val startTimestamp = LocalDate.parse(str1).toEpochDay * 86400 - 32400
    val endTimestamp = LocalDate.parse(str2).plusDays(1).toEpochDay * 86400 - 32400
    (startTimestamp, endTimestamp)
  }

  /**
    * Filter rows with null member IDs from a DataFrame.
    *
    * @param df A Spark DataFrame with a member ID column.
    * @return The filtered Spark DataFrame.
    */
  def filterNullMemberId(df: DataFrame): DataFrame = {
    df.filter(col(MemberId).isNotNull)
  }

  /**
    * Filter a DataFrame by an action type.
    *
    * @param actionType A action type to select.
    * @param df         A Spark DataFrame with an action type column.
    * @return The filtered Spark DataFrame.
    */
  def filterByActionType(actionType: String)(df: DataFrame): DataFrame = {
    if (actionType == "A") df
    else df.filter(col(ActionType) === actionType)
  }

  /**
    * Filter a DataFrame by starting and ending timestamps.
    *
    * @param startTimestamp The POSIX timestamp of a start date.
    * @param endTimestamp   The POSIX timestamp of a end date.
    * @param df             A Spark DataFrame with a timestamp column.
    * @return The filtered Spark DataFrame.
    */
  def filterByTimestamp(startTimestamp: Long, endTimestamp: Long)(df: DataFrame): DataFrame = {
    df.filter(col(Timestamp).between(startTimestamp, endTimestamp))
  }

  /**
    * Change member IDs in a DataFrame to uppercase.
    *
    * @param df A Spark DataFrame with a member ID column.
    * @return A Spark DataFrame with the uppercase IDs.
    */
  def upperMemberId(df: DataFrame): DataFrame = {
    df.withColumn(MemberId, upper(col(MemberId)))
  }

  /**
    * Join two Spark DataFrames by a column.
    *
    * @param right A Spark DataFrame to join from the right side.
    * @param on    The name of a column to join on.
    * @param left  A Spark DataFrame to join from the left side.
    * @return The joined Spark DataFrame.
    */
  def join(right: DataFrame, on: String)(left: DataFrame): DataFrame = {
    left.join(right, Seq(on), "inner")
  }

  /**
    * Remove redundant columns from a DataFrame.
    *
    * @param df A Spark DataFrame to remove columns.
    * @return The filtered Spark DataFrame.
    */
  def filterColumns(df: DataFrame): DataFrame = {
    df.select(BasicColumns.head, BasicColumns.tail: _*)
  }

  /**
    * Remove rows of a DataFrame, with products not reaching a frequency threshold.
    *
    * @param prodThreshold A product frequency threshold.
    * @param df            A Spark DataFrame with a product number column.
    * @return The filtered Spark DataFrame.
    */
  def filterByProdThreshold(prodThreshold: Int)(df: DataFrame): DataFrame = {
    val proper_prods = df
      .groupBy(ProductNo)
      .count()
      .filter(col("count") > prodThreshold)

    df.join(proper_prods, Seq(ProductNo), "left_semi")
  }

  /**
    * Sort rows of a DataFrame by a specific column.
    *
    * @param on The name of a column to sort with.
    * @param df A Spark DataFrame to sort rows.
    * @return The sorted DataFrame.
    */
  def sort(on: String)(df: DataFrame): DataFrame = {
    df.orderBy(on)
  }

}

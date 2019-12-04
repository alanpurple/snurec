/* Wemakeprice Recommendation Project.
 *
 * Authors:
 * - Hyunsik Jeon (jeon185@snu.ac.kr)
 * - Jaemin Yoo (jaeminyoo@snu.ac.kr)
 * - U Kang (ukang@snu.ac.kr)
 * - Data Mining Lab. at Seoul National University.
 *
 * File: wemakeprice/columns/package.scala
 * - Package object containing constants of DataFrame column names.
 *
 * Version: 1.0.0
 */
package wemakeprice

package object columns {
  val Timestamp = "timestamp"
  val ActionType = "action_type"
  val SearchKeyword = "search_keyword"

  val MemberId = "m_id"
  val CompMemberId = "comp_mid"
  val Age = "age"
  val Sex = "sex"

  val ProductNo = "prod_no"
  val Price = "sale_price"
  val SaleStartDate = "sale_start_dt"
  val SaleEndDate = "sale_end_dt"
  val ProductName = "prod_nm"
  val DealConstruction = "deal_construction"

  val Depth0Name = "gnb_depth0_nm"
  val Depth1No = "gnb_depth1_cate_id"
  val Depth1Name = "gnb_depth1_nm"
  val Depth2No = "gnb_depth2_cate_id"
  val Depth2Name = "gnb_depth2_nm"
  val Depth0DisplayOrder = "display_order_depth0"
  val Depth1DisplayOrder = "display_order_depth1"
  val Depth2DisplayOrder = "display_order_depth2"
  val CategoryNo = "gnb_category_id"

  val DealNo = "deal_no"
  val ItemNo = "item_no"

  val UseractionColumns: Seq[String] = Seq(
    Timestamp,
    ActionType,
    ProductNo,
    SearchKeyword,
    MemberId)

  val BasicColumns: Seq[String] = Seq(
    CompMemberId,
    Timestamp,
    ProductNo,
    Sex,
    Age,
    Price,
    SaleStartDate,
    SaleEndDate,
    ProductName,
    DealConstruction,
    CategoryNo,
    Depth0Name,
    Depth1No,
    Depth1Name,
    Depth2No,
    Depth2Name,
    Depth0DisplayOrder,
    Depth1DisplayOrder,
    Depth2DisplayOrder)
}

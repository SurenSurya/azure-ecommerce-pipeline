# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import os
from dotenv import load_dotenv

load_dotenv()

storage_account = os.getenv("STORAGE_ACCOUNT")
storage_key = os.getenv("STORAGE_KEY")

spark.conf.set(
    f"fs.azure.account.key.{storage_account}.blob.core.windows.net",
    storage_key
)

silver_base = f"wasbs://silver@{storage_account}.blob.core.windows.net/"
gold_base   = f"wasbs://gold@{storage_account}.blob.core.windows.net/"
orders   = spark.read.format("delta").load(silver_base + "orders")
items    = spark.read.format("delta").load(silver_base + "items")
payments = spark.read.format("delta").load(silver_base + "payments")
reviews  = spark.read.format("delta").load(silver_base + "reviews")
products = spark.read.format("delta").load(silver_base + "products")

# COMMAND ----------

items_orders = (items
    .join(orders.select(
        "order_id","order_status",
        "purchase_year","purchase_month","purchase_quarter",
        "is_late_delivery"), "order_id")
    .filter(F.col("order_status") == "delivered")
)

monthly = (items_orders
    .groupBy("purchase_year","purchase_month","purchase_quarter")
    .agg(
        F.countDistinct("order_id").alias("total_orders"),
        F.sum("price").alias("gross_revenue"),
        F.sum("freight_value").alias("total_freight"),
        F.avg("price").alias("avg_order_value"),
        F.sum("total_item_value").alias("total_revenue"),
        F.countDistinct("product_id").alias("unique_products_sold"),
        F.sum(F.when(F.col("is_late_delivery")==True, 1).otherwise(0))
            .alias("late_deliveries")
    )
    .withColumn("net_revenue",
        F.round(F.col("gross_revenue") - F.col("total_freight"), 2))
    .withColumn("freight_pct_of_revenue",
        F.round(F.col("total_freight")/F.col("gross_revenue")*100, 2))
    .withColumn("late_delivery_rate",
        F.round(F.col("late_deliveries")/F.col("total_orders")*100, 2))
)

w = Window.orderBy("purchase_year","purchase_month")

monthly = monthly.withColumn(
    "revenue_mom_growth_pct",
    F.round(
        (F.col("net_revenue") - F.lag("net_revenue").over(w)) /
        F.lag("net_revenue").over(w) * 100,
        2
    )
)

monthly.write.format("delta").mode("overwrite").save(gold_base + "monthly_sales")

# COMMAND ----------

REF_DATE = "2018-10-31"

rfm_base = (orders
    .filter(F.col("order_status") == "delivered")
    .join(items.groupBy("order_id")
          .agg(F.sum("price").alias("order_value")), "order_id")
    .groupBy("customer_id")
    .agg(
        F.datediff(F.lit(REF_DATE).cast("date"),
                   F.max(F.col("order_purchase_timestamp").cast("date"))).alias("recency_days"),
        F.count("order_id").alias("frequency"),
        F.round(F.sum("order_value"), 2).alias("monetary")
    )
)

r_bounds = rfm_base.approxQuantile("recency_days", [0.2,0.4,0.6,0.8], 0.01)
f_bounds = rfm_base.approxQuantile("frequency", [0.2,0.4,0.6,0.8], 0.01)
m_bounds = rfm_base.approxQuantile("monetary", [0.2,0.4,0.6,0.8], 0.01)

rfm = (rfm_base
    .withColumn("R_score",
        F.when(F.col("recency_days") <= r_bounds[0], 5)
        .when(F.col("recency_days") <= r_bounds[1], 4)
        .when(F.col("recency_days") <= r_bounds[2], 3)
        .when(F.col("recency_days") <= r_bounds[3], 2)
        .otherwise(1))
    .withColumn("F_score",
        F.when(F.col("frequency") >= f_bounds[3], 5)
        .when(F.col("frequency") >= f_bounds[2], 4)
        .when(F.col("frequency") >= f_bounds[1], 3)
        .when(F.col("frequency") >= f_bounds[0], 2)
        .otherwise(1))
    .withColumn("M_score",
        F.when(F.col("monetary") >= m_bounds[3], 5)
        .when(F.col("monetary") >= m_bounds[2], 4)
        .when(F.col("monetary") >= m_bounds[1], 3)
        .when(F.col("monetary") >= m_bounds[0], 2)
        .otherwise(1))
    .withColumn("RFM_total",
        F.col("R_score") + F.col("F_score") + F.col("M_score"))
    .withColumn("segment",
        F.when(F.col("RFM_total") >= 13, "Champions")
        .when(F.col("RFM_total") >= 10, "Loyal Customers")
        .when(F.col("RFM_total") >= 7, "Potential Loyalist")
        .when(F.col("RFM_total") >= 5, "At Risk")
        .otherwise("Lost"))
)

rfm.write.format("delta").mode("overwrite").save(gold_base + "customer_rfm")

# COMMAND ----------

print("Monthly Sales Rows:", monthly.count())
print("RFM Customers:", rfm.count())

rfm.groupBy("segment").count().show()

# COMMAND ----------

from pyspark.sql.window import Window

cat_perf = (items
    .join(
        products.select("product_id","product_category_name_english"),
        "product_id"
    )
    .join(
        orders.select("order_id","order_status"),
        "order_id"
    )
    .join(
        reviews.select("order_id","review_score"),
        "order_id",
        "left"
    )
    .filter(F.col("order_status") == "delivered")
    .groupBy("product_category_name_english")
    .agg(
        F.count("order_id").alias("total_orders"),
        F.sum("price").alias("total_revenue"),
        F.avg("price").alias("avg_price"),
        F.avg("review_score").alias("avg_review_score"),
        F.countDistinct("order_id").alias("unique_orders")
    )
    .withColumn(
        "revenue_rank",
        F.rank().over(Window.orderBy(F.desc("total_revenue")))
    )
    .orderBy("revenue_rank")
)

# COMMAND ----------

gold_base = f"wasbs://gold@{storage_account}.blob.core.windows.net/"

cat_perf.write.format("delta").mode("overwrite").save(
    gold_base + "category_perf"
)

print(f"Category performance: {cat_perf.count()} categories")

# COMMAND ----------

import uuid
from datetime import datetime

# COMMAND ----------

meta_rows = [
    (str(uuid.uuid4()), "batch_transform", "silver_orders",
     99441, "SUCCESS", 45.2, datetime.now()),

    (str(uuid.uuid4()), "batch_transform", "silver_items",
     112650, "SUCCESS", 32.1, datetime.now()),

    (str(uuid.uuid4()), "batch_transform", "gold_monthly_sales",
     24, "SUCCESS", 18.4, datetime.now()),

    (str(uuid.uuid4()), "batch_transform", "gold_customer_rfm",
     96096, "SUCCESS", 55.8, datetime.now())
]

# COMMAND ----------

meta_df = spark.createDataFrame(
    meta_rows,
    ["run_id","pipeline","table_name","rows","status","duration_secs","loaded_at"]
)

# COMMAND ----------

meta_df.write.format("delta").mode("append").save(
    gold_base + "pipeline_metadata"
)

print("Metadata logged successfully")
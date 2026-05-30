# Databricks notebook source
from pyspark.sql import functions as F
from pyspark.sql.types import *
import time

# COMMAND ----------

start = time.time()

# COMMAND ----------
import os
from dotenv import load_dotenv

load_dotenv()

storage_account = os.getenv("STORAGE_ACCOUNT")
storage_key = os.getenv("STORAGE_KEY")

if not storage_account or not storage_key:
    raise ValueError("Storage configuration variables are missing from environment.")

spark.conf.set(
    f"fs.azure.account.key.{storage_account}.blob.core.windows.net",
    storage_key
)
print("ADLS connection configured successfully")

# COMMAND ----------

bronze_path = f"wasbs://bronze@{storage_account}.blob.core.windows.net/olist/"

# COMMAND ----------

raw_orders = (spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("multiLine", "true")
    .csv(bronze_path + "olist_orders_dataset*.csv")
)

print(f"Raw rows: {raw_orders.count():,}")

# COMMAND ----------

orders = (raw_orders
    .withColumn("order_purchase_timestamp",
        F.to_timestamp("order_purchase_timestamp", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("order_approved_at",
        F.to_timestamp("order_approved_at", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("order_delivered_carrier_date",
        F.to_timestamp("order_delivered_carrier_date", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("order_delivered_customer_date",
        F.to_timestamp("order_delivered_customer_date", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("order_estimated_delivery_date",
        F.to_timestamp("order_estimated_delivery_date", "yyyy-MM-dd HH:mm:ss"))
)

# COMMAND ----------

orders = (orders
    .withColumn("purchase_year", F.year("order_purchase_timestamp"))
    .withColumn("purchase_month", F.month("order_purchase_timestamp"))
    .withColumn("purchase_quarter", F.quarter("order_purchase_timestamp"))
    .withColumn("purchase_dow", F.dayofweek("order_purchase_timestamp"))
    .withColumn("purchase_hour", F.hour("order_purchase_timestamp"))
    .withColumn("is_weekend",
        F.when(F.dayofweek("order_purchase_timestamp").isin(1,7), True)
        .otherwise(False))
)

# COMMAND ----------

orders = (orders
    .withColumn("approval_time_hrs",
        F.round(
            (F.unix_timestamp("order_approved_at") -
             F.unix_timestamp("order_purchase_timestamp")) / 3600, 2))
    .withColumn("delivery_delay_days",
        F.when(F.col("order_delivered_customer_date").isNotNull(),
            F.datediff("order_delivered_customer_date",
                       "order_estimated_delivery_date"))
        .otherwise(None))
    .withColumn("is_late_delivery",
        F.when(F.col("delivery_delay_days") > 0, True)
        .when(F.col("delivery_delay_days").isNull(), None)
        .otherwise(False))
)

# COMMAND ----------

orders = orders.dropna(subset=["order_id", "customer_id", "order_status"])

# COMMAND ----------

orders = (orders
    .withColumn("ingested_at", F.current_timestamp())
    .withColumn("source_file", F.lit("olist_orders_dataset"))
)

# COMMAND ----------

silver_path = f"wasbs://silver@{storage_account}.blob.core.windows.net/orders/"

# COMMAND ----------

(orders.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("purchase_year", "purchase_month")
    .save(silver_path)
)

# COMMAND ----------

duration = round(time.time() - start, 1)
print(f"Silver orders written: {orders.count():,} rows in {duration}s")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

items_raw = spark.read.option("header","true").csv(
    f"wasbs://bronze@{storage_account}.blob.core.windows.net/olist/olist_order_items_dataset*.csv"
)

items = (items_raw
    .withColumn("price", F.col("price").cast(DoubleType()))
    .withColumn("freight_value", F.col("freight_value").cast(DoubleType()))
    .withColumn("shipping_limit_date",
        F.to_timestamp("shipping_limit_date", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("total_item_value",
        F.round(F.col("price") + F.col("freight_value"), 2))
    .withColumn("freight_pct",
        F.round(F.col("freight_value") / F.col("price") * 100, 2))
    .dropna(subset=["order_id","product_id","price"])
)

items.write.format("delta").mode("overwrite").save(
    f"wasbs://silver@{storage_account}.blob.core.windows.net/items"
)

print(f"Silver items: {items.count():,}")

# COMMAND ----------

pay_raw = spark.read.option("header","true").csv(
    f"wasbs://bronze@{storage_account}.blob.core.windows.net/olist/olist_order_payments_dataset*.csv"
)

payments = (pay_raw
    .withColumn("payment_value",
        F.col("payment_value").cast(DoubleType()))
    .withColumn("payment_installments",
        F.col("payment_installments").cast(IntegerType()))
    .withColumn("is_credit_card",
        F.when(F.col("payment_type")=="credit_card", True).otherwise(False))
    .dropna(subset=["order_id","payment_value"])
)

payments.write.format("delta").mode("overwrite").save(
    f"wasbs://silver@{storage_account}.blob.core.windows.net/payments"
)

print(f"Silver payments: {payments.count():,}")

# COMMAND ----------

rev_raw = spark.read.option("header","true").csv(
    f"wasbs://bronze@{storage_account}.blob.core.windows.net/olist/olist_order_reviews_dataset*.csv"
)

reviews = (rev_raw
    .withColumn("review_score",
        F.col("review_score").cast(IntegerType()))
    .withColumn("has_text_comment",
        F.when(
            F.col("review_comment_message").isNotNull() &
            (F.trim(F.col("review_comment_message")) != ""),
            True
        ).otherwise(False))
    .withColumn("review_creation_date",
        F.to_timestamp("review_creation_date","yyyy-MM-dd HH:mm:ss"))
    .dropna(subset=["order_id","review_score"])
)

reviews.write.format("delta").mode("overwrite").save(
    f"wasbs://silver@{storage_account}.blob.core.windows.net/reviews"
)

print(f"Silver reviews: {reviews.count():,}")

# COMMAND ----------

prod_raw = spark.read.option("header","true").csv(
    f"wasbs://bronze@{storage_account}.blob.core.windows.net/olist/olist_products_dataset*.csv"
)

trans = spark.read.option("header","true").csv(
    f"wasbs://bronze@{storage_account}.blob.core.windows.net/olist/product_category_name_translation*.csv"
)

products = (prod_raw
    .join(trans, "product_category_name", "left")
    .withColumn("product_weight_g",
        F.col("product_weight_g").cast(IntegerType()))
    .withColumn("product_photos_qty",
        F.col("product_photos_qty").cast(IntegerType()))
    .fillna({"product_category_name_english": "uncategorized"})
)

products.write.format("delta").mode("overwrite").save(
    f"wasbs://silver@{storage_account}.blob.core.windows.net/products"
)

print(f"Silver products: {products.count():,}")
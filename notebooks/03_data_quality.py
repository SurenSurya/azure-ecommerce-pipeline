# Databricks notebook source
from pyspark.sql import functions as F

results = []

def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append({"check": name, "status": status, "detail": str(detail)})
    print(f"[{status}] {name}: {detail}")

# COMMAND ----------

# DBTITLE 1,Cell 2
from dotenv import load_dotenv
import os

load_dotenv()

storage_account = os.getenv("STORAGE_ACCOUNT")
storage_key = os.getenv("STORAGE_KEY")

spark.conf.set(
    f"fs.azure.account.key.{storage_account}.blob.core.windows.net",
    storage_key
)

silver_base = f"wasbs://silver@{storage_account}.blob.core.windows.net/"
gold_base   = f"wasbs://gold@{storage_account}.blob.core.windows.net/"

orders  = spark.read.format("delta").load(silver_base + "orders")
items   = spark.read.format("delta").load(silver_base + "items")

rfm     = spark.read.format("delta").load(gold_base + "customer_rfm")
monthly = spark.read.format("delta").load(gold_base + "monthly_sales")

# COMMAND ----------

check("orders_count", orders.count() > 90000, f"{orders.count():,} rows")
check("items_count",  items.count() > 100000, f"{items.count():,} rows")
check("rfm_count",    rfm.count() > 80000,    f"{rfm.count():,} customers")

# COMMAND ----------

null_oids = orders.filter(F.col("order_id").isNull()).count()

check(
    "no_null_order_id",
    null_oids == 0,
    f"{null_oids} nulls"
)

# COMMAND ----------

null_oids = orders.filter(F.col("order_id").isNull()).count()

check(
    "no_null_order_id",
    null_oids == 0,
    f"{null_oids} nulls"
)

# COMMAND ----------

segs = set(r[0] for r in rfm.select("segment").distinct().collect())

expected_segs = {
    "Champions",
    "Loyal Customers",
    "Potential Loyalist",
    "At Risk",
    "Lost"
}

check(
    "rfm_segments_complete",
    segs == expected_segs,
    segs
)

# COMMAND ----------

neg_rev = monthly.filter(F.col("net_revenue") < 0).count()

check(
    "no_negative_revenue",
    neg_rev == 0,
    f"{neg_rev} negative rows"
)

# COMMAND ----------

min_date = orders.agg(F.min("order_purchase_timestamp")).collect()[0][0]
max_date = orders.agg(F.max("order_purchase_timestamp")).collect()[0][0]

check(
    "date_range_valid",
    "2016" in str(min_date) and "2018" in str(max_date),
    f"{min_date} to {max_date}"
)

# COMMAND ----------

missing = expected_segs - segs

check(
    "rfm_segments_complete",
    len(missing) == 0,
    f"Missing: {missing}"
)
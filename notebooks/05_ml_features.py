# Databricks notebook source
# DBTITLE 1,Cell 1
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

monthly = spark.read.format("delta").load(
    f"wasbs://gold@{storage_account}.blob.core.windows.net/monthly_sales"
)
# COMMAND ----------

# DBTITLE 1,Cell 2
features = (monthly
    .withColumn(
        "period_date",
        F.to_date(
            F.concat_ws(
                "-",
                F.col("purchase_year").cast("string"),
                F.lpad(F.col("purchase_month").cast("string"), 2, "0"),
                F.lit("01")
            )
        )
    )
    .orderBy("period_date")
)

# COMMAND ----------

w = Window.orderBy("period_date")

# COMMAND ----------

features = (features
    .withColumn("revenue_lag1", F.lag("net_revenue", 1).over(w))
    .withColumn("revenue_lag2", F.lag("net_revenue", 2).over(w))
    .withColumn("revenue_lag3", F.lag("net_revenue", 3).over(w))
)

# COMMAND ----------

features = (features
    .withColumn("rolling_3m_avg",
        F.avg("net_revenue").over(w.rowsBetween(-3, -1)))
    .withColumn("rolling_3m_std",
        F.stddev("net_revenue").over(w.rowsBetween(-3, -1)))
)

# COMMAND ----------

features = (features
    .withColumn("orders_lag1", F.lag("total_orders", 1).over(w))
    .withColumn("month_sin", F.sin(F.col("purchase_month") * 3.14159 * 2 / 12))
    .withColumn("month_cos", F.cos(F.col("purchase_month") * 3.14159 * 2 / 12))
    .withColumn("is_q4", F.when(F.col("purchase_quarter") == 4, 1).otherwise(0))
    .withColumn("is_year_end", F.when(F.col("purchase_month") == 12, 1).otherwise(0))
)

# COMMAND ----------

features = (features
    .withColumn("prev_mom_growth", F.lag("revenue_mom_growth_pct", 1).over(w))
    .dropna()
)

# COMMAND ----------

features.write.format("delta").mode("overwrite").option("mergeSchema", "true").save(f"wasbs://gold@{storage_account}.blob.core.windows.net/monthly_sales")

print(f"Feature rows: {features.count()}")
display(features.select(
    "period_date", "net_revenue", "revenue_lag1",
    "rolling_3m_avg", "is_q4"
).orderBy("period_date"))
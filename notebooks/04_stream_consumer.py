# Databricks notebook source
dbutils.widgets.text("eh_conn_str", "")
EH_CONN = dbutils.widgets.get("eh_conn_str")

EVENT_HUB_NAME = "eh-orders-stream"

if not EH_CONN:
    raise ValueError("Please paste Event Hub connection string")

# ADD EntityPath if missing
if "EntityPath" not in EH_CONN:
    EH_CONN = EH_CONN + f";EntityPath={EVENT_HUB_NAME}"

# COMMAND ----------

eh_conf = {
    "eventhubs.connectionString":
        sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(EH_CONN)
}

# COMMAND ----------

raw_stream = (spark.readStream
    .format("eventhubs")
    .options(**eh_conf)
    .load())

# COMMAND ----------

event_schema = StructType([
    StructField("event_id", StringType()),
    StructField("order_id", StringType()),
    StructField("customer_id", StringType()),
    StructField("status", StringType()),
    StructField("purchase_ts", StringType()),
    StructField("item_count", IntegerType()),
    StructField("total_value", DoubleType()),
    StructField("event_time", StringType())
])

# COMMAND ----------

parsed = (raw_stream
    .withColumn("body", F.col("body").cast("string"))
    .withColumn("data", F.from_json("body", event_schema))
    .select(
        "data.*",
        F.col("enqueuedTime").alias("eh_enqueued_time"),
        F.current_timestamp().alias("processing_time")
    )
    .withColumn("event_time", F.to_timestamp("event_time"))
)

# COMMAND ----------

# DBTITLE 1,Cell 6
raw_query = (parsed.writeStream
    .format("delta")
    .outputMode("append")
    .option(
        "checkpointLocation",
        "wasbs://stream-landing@ecomdatalakesurya.blob.core.windows.net/checkpoints/raw"
    )
    .trigger(processingTime="30 seconds")
    .start(
        "wasbs://stream-landing@ecomdatalakesurya.blob.core.windows.net/raw_events"
    ))

# COMMAND ----------

windowed_agg = (parsed
    .withWatermark("event_time", "10 minutes")
    .groupBy(
        F.window("event_time", "5 minutes", "1 minute"),
        "status"
    )
    .agg(
        F.count("order_id").alias("order_count"),
        F.sum("total_value").alias("window_revenue"),
        F.avg("total_value").alias("avg_order_value"),
        F.sum("item_count").alias("total_items")
    )
    .withColumn("window_start", F.col("window.start"))
    .withColumn("window_end", F.col("window.end"))
    .drop("window")
)

# COMMAND ----------

agg_query = (windowed_agg.writeStream
    .format("delta")
    .outputMode("append")
    .option(
        "checkpointLocation",
        "wasbs://stream-landing@ecomdatalakesurya.blob.core.windows.net/checkpoints/agg"
    )
    .trigger(processingTime="60 seconds")
    .start(
        "wasbs://stream-landing@ecomdatalakesurya.blob.core.windows.net/windowed_agg"
    ))

# COMMAND ----------

print("Streaming started successfully")
print(f"Active queries: {len(spark.streams.active)}")

# COMMAND ----------

for q in spark.streams.active:
    print(f"Query: {q.name or 'unnamed'}")
    print(f"Status: {q.status}")

    lp = q.lastProgress
    if lp:
        print(f"  Input rows/sec: {lp.get('inputRowsPerSecond', 0):.1f}")
        print(f"  Processed rows/sec: {lp.get('processedRowsPerSecond', 0):.1f}")

    print("-" * 40)

# COMMAND ----------

raw_count = spark.read.format("delta").load(
    "wasbs://stream-landing@ecomdatalakesurya.blob.core.windows.net/raw_events"
).count()

agg_count = spark.read.format("delta").load(
    "wasbs://stream-landing@ecomdatalakesurya.blob.core.windows.net/windowed_agg"
).count()

print(f"Raw events landed: {raw_count:,}")
print(f"Windowed aggregates: {agg_count:,}")

# COMMAND ----------

for q in spark.streams.active:
    print(f"Stopping: {q.name or 'unnamed'}")
    q.stop()

print("All streams stopped successfully")
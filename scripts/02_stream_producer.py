import os, json, time, pandas as pd
from azure.eventhub import EventHubProducerClient, EventData
from dotenv import load_dotenv

load_dotenv()
CONN_STR = os.environ["EVENTHUB_CONNECTION_STRING"]
EH_NAME  = os.environ["EVENTHUB_NAME"]

# Load source data
orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
items  = pd.read_csv("data/raw/olist_order_items_dataset.csv")

# Pre-compute item totals per order (faster than repeated filtering)
order_totals = (items.groupby("order_id")
    .agg(item_count=("order_id","count"),
         total_value=("price","sum"))
    .reset_index())

merged = orders.merge(order_totals, on="order_id", how="left")
merged["item_count"]  = merged["item_count"].fillna(0).astype(int)
merged["total_value"] = merged["total_value"].fillna(0.0)

print(f"Streaming {len(merged):,} order events to Event Hubs...")
print(f"Namespace: {CONN_STR.split(';')[0]}")

producer = EventHubProducerClient.from_connection_string(
    conn_str=CONN_STR, eventhub_name=EH_NAME)

sent = 0
with producer:
    for _, row in merged.iterrows():
        payload = {
            "event_id":    f"evt_{row['order_id'][:8]}",
            "order_id":    row["order_id"],
            "customer_id": row["customer_id"],
            "status":      row["order_status"],
            "purchase_ts": str(row["order_purchase_timestamp"]),
            "item_count":  int(row["item_count"]),
            "total_value": round(float(row["total_value"]), 2),
            "event_time":  pd.Timestamp.now().isoformat()
        }
        batch = producer.create_batch()
        batch.add(EventData(json.dumps(payload)))
        producer.send_batch(batch)
        sent += 1
        if sent % 100 == 0:
            print(f"Sent {sent:,} events...")
        time.sleep(0.3)   # ~3 events/sec — adjust to control throughput

print(f"Done. Total sent: {sent:,} events")
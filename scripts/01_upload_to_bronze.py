import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Azure Storage details
ACCOUNT   = os.environ["AZURE_STORAGE_ACCOUNT"]
ACCT_KEY  = os.environ["AZURE_STORAGE_KEY"]
CONTAINER = "bronze"

# Build connection string
conn_str = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={ACCOUNT};"
    f"AccountKey={ACCT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# Create Blob client
client = BlobServiceClient.from_connection_string(conn_str)
container_client = client.get_container_client(CONTAINER)

# List of CSV files to upload
csv_files = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv"
]

# Upload files
for fname in csv_files:

    local_path = f"data/raw/{fname}"
    blob_path  = f"olist/{fname}"

    print(f"Uploading {fname}...")

    try:
        with open(local_path, "rb") as f:

            container_client.upload_blob(
                name=blob_path,
                data=f,
                overwrite=True,
                timeout=600,
                max_concurrency=4
            )

        print(f"  Done → bronze/olist/{fname}")

    except Exception as e:
        print(f"  Failed to upload {fname}")
        print(f"  Error: {e}")

print("All upload attempts completed")
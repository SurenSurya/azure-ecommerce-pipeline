# Databricks notebook source
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install --upgrade typing_extensions python-dotenv

# COMMAND ----------

# Load environment variables from the .env file
load_dotenv()

storage_account = os.getenv("STORAGE_ACCOUNT")
storage_key = os.getenv("STORAGE_KEY")
experiment_path = os.getenv("MLFLOW_EXPERIMENT_PATH", "/Shared/ecommerce-sales-forecast")
model_name = os.getenv("MODEL_REGISTRY_NAME", "adb_ecom_pipeline.default.ecommerce-revenue-forecast")

# Basic validation to ensure pipeline secrets are loaded
if not storage_account or not storage_key:
    raise ValueError("Missing critical environment variables: STORAGE_ACCOUNT or STORAGE_KEY.")

# Set MLflow experiment dynamically via env file path
mlflow.set_experiment(experiment_path)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Authenticate with the Storage Account using environment variables
spark.conf.set(
    f"fs.azure.account.key.{storage_account}.blob.core.windows.net",
    storage_key
)

# Load features from the gold container dynamically
features_pd = (spark.read.format("delta")
    .load(f"wasbs://gold@{storage_account}.blob.core.windows.net/monthly_sales")
    .orderBy("period_date")
    .toPandas())

# COMMAND ----------

FEATURE_COLS = [
    "revenue_lag1", "revenue_lag2", "revenue_lag3",
    "rolling_3m_avg", "rolling_3m_std",
    "orders_lag1", "month_sin", "month_cos",
    "is_q4", "is_year_end", "prev_mom_growth"
]
TARGET_COL = "net_revenue"

# Drop rows with null values resulting from lag/rolling window generation
features_pd = features_pd.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

X = features_pd[FEATURE_COLS]
y = features_pd[TARGET_COL]

# COMMAND ----------

tscv = TimeSeriesSplit(n_splits=3)

# Define standard search spaces for candidate algorithms
models_to_evaluate = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

best_model_name = None
best_model_obj = None
best_r2 = -float("inf")
best_run_id = None

for algo_name, model in models_to_evaluate.items():
    with mlflow.start_run(run_name=f"CV_{algo_name}") as run:
        mlflow.log_param("algorithm", algo_name)
        
        cv_maes, cv_mses, cv_r2s = [], [], []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train the current model architecture on fold splits
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            # Evaluate standard metrics
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            cv_maes.append(mae)
            cv_mses.append(mse)
            cv_r2s.append(r2)
            
            mlflow.log_metric(f"fold_{fold}_mae", mae)
            mlflow.log_metric(f"fold_{fold}_r2", r2)
            
        avg_mae = np.mean(cv_maes)
        avg_mse = np.mean(cv_mses)
        avg_r2 = np.mean(cv_r2s)
        
        mlflow.log_metric("avg_mae", avg_mae)
        mlflow.log_metric("avg_mse", avg_mse)
        mlflow.log_metric("avg_r2", avg_r2)
        
        # Track the dominant model based on R-Squared metrics
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            best_model_name = algo_name
            best_model_obj = model
            best_run_id = run.info.run_id

print(f"Best cross-validated algorithm architecture: {best_model_name} with average R2: {best_r2:.4f}")

# COMMAND ----------

# Train the finalized model candidate on all available feature data
final_model = models_to_evaluate[best_model_name]
final_model.fit(X, y)

with mlflow.start_run(run_name="Final_Production_Model") as run:
    mlflow.log_param("chosen_algorithm", best_model_name)
    
    # Generate unified execution signatures for input tracking
    input_example = X.head(3)
    signature = mlflow.models.infer_signature(X, final_model.predict(X))
    
    # Log model artifacts into MLflow tracking servers
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )
    
    final_run_id = run.info.run_id
    print(f"Final production model artifact registered into Run ID: {final_run_id}")

# COMMAND ----------

# Register the model using the model registry variable from the environment
model_uri = f"runs:/{final_run_id}/model"
result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(f"Model successfully processed within registry path: {model_name}")

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Query the model registry dynamically using variables
versions = client.search_model_versions(f"name='{model_name}'")
latest_version = max([int(v.version) for v in versions])

print(f"Extracted latest deployed version tag inside registry: Version {latest_version}")

# Assign Aliases to transition versioning states cleanly
client.set_registered_model_alias(
    name=model_name,
    alias="Challenger",
    version=str(latest_version)
)
print(f"Assigned 'Challenger' routing status target onto Version {latest_version}")

# COMMAND ----------

# Fetch Challenger model configuration targets directly from the registry
model_challenger_uri = f"models:/{model_name}@Challenger"
model = mlflow.sklearn.load_model(model_challenger_uri)

# COMMAND ----------

last = features_pd.iloc[-1]
history_rev = features_pd["net_revenue"].tolist()

future_rows = []
for i in range(1, 4):
    month_num = int((last["purchase_month"] + i - 1) % 12 + 1)
    
    row = {
        "revenue_lag1": history_rev[-1],
        "revenue_lag2": history_rev[-2],
        "revenue_lag3": history_rev[-3],
        "rolling_3m_avg": np.mean(history_rev[-3:]),
        "rolling_3m_std": np.std(history_rev[-3:]),
        "orders_lag1": last["orders_lag1"],
        "month_sin": np.sin(month_num * 3.14159 * 2 / 12),
        "month_cos": np.cos(month_num * 3.14159 * 2 / 12),
        "is_q4": 1 if month_num in [10, 11, 12] else 0,
        "is_year_end": 1 if month_num == 12 else 0,
        "prev_mom_growth": last["revenue_mom_growth_pct"]
    }
    future_rows.append(row)

future_df = pd.DataFrame(future_rows)

# Fix datatype mismatches to match tracked MLflow validation schemas
future_df["is_q4"] = future_df["is_q4"].astype("int32")
future_df["is_year_end"] = future_df["is_year_end"].astype("int32")

future_df["predicted_revenue"] = model.predict(future_df[FEATURE_COLS])
future_df["forecast_horizon"] = [1, 2, 3]
future_df["model_name"] = best_model_name
future_df["forecast_date"] = pd.Timestamp.now()

# Convert inferences back into Spark dataframes and export to your gold layer
forecast_spark = spark.createDataFrame(future_df)
forecast_spark.write.format("delta").mode("append").save(
    f"wasbs://gold@{storage_account}.blob.core.windows.net/revenue_forecasts"
)

print("Inference forecasts saved successfully.")
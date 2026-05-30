# 🚀 Azure E-Commerce Analytics Platform (End-to-End Data Engineering + ML)

## 📌 Project Overview

This project is a complete cloud-based data engineering and analytics platform built on Microsoft Azure. It simulates a real-world e-commerce ecosystem with batch ingestion, real-time streaming, data lake architecture (Bronze–Silver–Gold), machine learning forecasting, and BI dashboards.

The system demonstrates how raw transactional data is transformed into business-ready insights and predictive analytics using modern data engineering practices.

---

## 🏗️ Architecture Overview

![Architecture](docs/architecture.png)

### Data Flow:
1. **Data Ingestion (Batch + Streaming)**
   - Azure Data Factory (ADF) for batch ingestion
   - Event Hub producer for real-time events

2. **Data Lake Layers (Medallion Architecture)**
   - Bronze → Raw ingestion (ADLS Gen2)
   - Silver → Cleaned & transformed data (Databricks PySpark)
   - Gold → Aggregated business-ready tables

3. **Streaming Pipeline**
   - Spark Structured Streaming
   - Windowed aggregations
   - Near real-time Delta Lake updates

4. **Machine Learning Layer**
   - Feature engineering in PySpark
   - Model training using Scikit-learn (Gradient Boosting)
   - Experiment tracking with MLflow
   - Forecasting demand/revenue trends

5. **Serving Layer**
   - Synapse Serverless SQL for querying Gold layer
   - Power BI dashboards for business insights

---

## ⚙️ Tech Stack

| Layer | Tools Used |
|------|------------|
| Cloud Platform | Microsoft Azure |
| Data Ingestion | Azure Data Factory, Event Hubs |
| Storage | Azure Data Lake Storage Gen2 |
| Processing | Databricks (PySpark, Delta Lake) |
| Streaming | Spark Structured Streaming |
| ML | Python, Scikit-learn, MLflow |
| Query Engine | Synapse Serverless SQL |
| Visualization | Power BI |
| Version Control | Git, GitHub |

---

## 📂 Repository Structure
azure-ecommerce-pipeline/
│
├── notebooks/
├── scripts/
├── infrastructure/
├── configs/
├── docs/
├── src/
├── requirements.txt
├── .env.example
└── README.md


---

## 🔄 Pipeline Breakdown

### Bronze Layer
- Raw CSV/JSON ingestion into ADLS Gen2
- ADF pipelines orchestrate batch loads
- Event Hub captures real-time transactions

### Silver Layer
- PySpark transformations in Databricks
- Data quality checks, null handling, deduplication
- Delta Lake schema enforcement

### Gold Layer
- Aggregated business metrics:
  - Revenue by category
  - Customer segmentation (RFM)
  - Sales trends

---

## ⚡ Streaming Pipeline

- Event Hub → Spark Structured Streaming → Delta Lake
- Window-based aggregations (5 min / 1 hour)
- Near real-time analytics

---

## 🤖 Machine Learning Pipeline

### Objective:
Forecast future revenue trends using historical sales data.

### Steps:
- Feature engineering (time-based + customer features)
- Gradient Boosting Regressor model
- MLflow experiment tracking
- Model registration

---

## 📊 Power BI Dashboard

- Sales Performance Dashboard
- Customer RFM Analysis
- Streaming Metrics
- ML Forecast Insights

---

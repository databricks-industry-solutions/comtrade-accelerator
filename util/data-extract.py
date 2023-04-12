# Databricks notebook source
# MAGIC %md The purpose of this notebook is to download and set up the data we will use for the solution accelerator. 

# COMMAND ----------

display(dbutils.fs.ls("s3://db-gtm-industry-solutions/data/rcg/comtrade/"))

# COMMAND ----------

dbutils.fs.cp("file:/databricks/driver/campaign_desc.csv", "dbfs:/tmp/propensity/bronze/campaign_desc.csv")

# COMMAND ----------

# MAGIC %md Move the downloaded data to the folder used throughout the accelerator:

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/campaign_desc.csv", "dbfs:/tmp/propensity/bronze/campaign_desc.csv")
dbutils.fs.mv("file:/databricks/driver/campaign_table.csv", "dbfs:/tmp/propensity/bronze/campaign_table.csv")
dbutils.fs.mv("file:/databricks/driver/causal_data.csv", "dbfs:/tmp/propensity/bronze/causal_data.csv")
dbutils.fs.mv("file:/databricks/driver/coupon.csv", "dbfs:/tmp/propensity/bronze/coupon.csv")
dbutils.fs.mv("file:/databricks/driver/coupon_redempt.csv", "dbfs:/tmp/propensity/bronze/coupon_redempt.csv")
dbutils.fs.mv("file:/databricks/driver/hh_demographic.csv", "dbfs:/tmp/propensity/bronze/hh_demographic.csv")
dbutils.fs.mv("file:/databricks/driver/product.csv", "dbfs:/tmp/propensity/bronze/product.csv")
dbutils.fs.mv("file:/databricks/driver/transaction_data.csv", "dbfs:/tmp/propensity/bronze/transaction_data.csv")

# COMMAND ----------



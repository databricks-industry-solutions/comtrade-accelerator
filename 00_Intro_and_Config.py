# Databricks notebook source
# MAGIC %md The purpose of this notebook is to introduce the COMTRADE solution accelerator and to provide access to configuration information for the notebooks supporting it.  This notebook was developed on a **Databricks ML 12.2 LTS** cluster. This notebook is also available at https://github.com/databricks-industry-solutions/comtrade-accelerator.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC In this solution accelerator, we will show how electric signal data captured in the COMTRADE format can be read using a Databricks cluster.  To help illustrate the potential of using Databricks for this work (besides its ability to process a large volume of data a timely manner through the power of distributed processing), we will use the data read from COMTRADE format to standard table objects to train a fault detection model.

# COMMAND ----------

# MAGIC %md ##Configuration

# COMMAND ----------

# DBTITLE 1,Instantiate Config Variable
if 'config' not in locals().keys():
  config = {}

# COMMAND ----------

# DBTITLE 1,Database
config['database'] = 'comtrade'

# create database if not exists
_ = spark.sql('create database if not exists {0}'.format(config['database']))

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# DBTITLE 1,File Paths
config['temp_point'] ='/tmp/comtrade'
config['source_path'] = "s3://db-gtm-industry-solutions/data/rcg/comtrade/source/"
config['input_path'] = config['temp_point'] + '/input'
config['output_path'] = config['temp_point'] + '/output'
dbutils.fs.mkdirs(config['input_path'])
dbutils.fs.mkdirs(config['output_path'])

# COMMAND ----------

# DBTITLE 1,Model
config['model_name'] = 'fault_detection'

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | comtrade | A module designed to read Common Format for Transient Data Exchange (COMTRADE) file format |  MIT | https://pypi.org/project/comtrade/                       |
# MAGIC | comtradehandlers | File handlers for the COMTRADE format| MIT | https://github.com/relihanl/comtradehandlers.git#egg=comtradehandlers |

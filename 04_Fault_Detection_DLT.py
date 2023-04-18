# Databricks notebook source
# MAGIC %pip install comtrade orjson mlflow tensorflow

# COMMAND ----------

# MAGIC %md ## Deploy the Model
# MAGIC 
# MAGIC 
# MAGIC With our model in MLflow, it can easily be deployed in a variety of scenarios.  The one that most immediately comes to mind is one within which COMTRADE data are delivered from electrical providers as part of their fault management processes.  These files may be processed upon receipt in real-time using Databricks [Auto Loader](https://docs.databricks.com/ingestion/auto-loader/index.html) and [Delta Live Table](https://docs.databricks.com/delta-live-tables/index.html) logic that persists the data to Delta Lake tables and [presented to the latest production instance of our fault detection model](https://docs.databricks.com/delta-live-tables/transform.html#use-mlflow-models-in-a-delta-live-tables-pipeline) to determine if a fault has occurred. From there, Databricks may send a message to any number of [downstream systems](https://docs.databricks.com/external-data/index.html) in order to notify them of the occurrence.
# MAGIC </p>
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/comtrade_architecture2.png" width=60%>
# MAGIC </p>
# MAGIC 
# MAGIC In this notebook, we combine all the ETL steps from notebook 02 with the ML model trained in notebook 03 to build a DLT pipeline for end-to-end model inference. 
# MAGIC **Do not run this notebook interactively** - it will fail on the block below on `import dlt`. This notebook would only run as part of a DLT pipeline. 
# MAGIC 
# MAGIC See the end of this notebook for instructions to set up the DLT pipeline using UI. Alternatively, if you use the RUNME notebook to create a Workflow for this accelerator, the DLT pipeline will be created for you as the last step in the Workflow. 

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports & Constants

# COMMAND ----------

import pyspark.sql.functions as F
import comtrade
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql import DataFrame
from pyspark.sql.types import BinaryType, StringType, StructType, StructField, ArrayType, DoubleType, MapType, TimestampType, IntegerType, LongType
import pandas as pd
import io
import pickle
import numpy as np
import datetime
from typing import Iterator, List
import orjson
import matplotlib.pyplot as plt
import dlt
import mlflow
import tensorflow
import mlflow.keras as ml_keras
import json

# COMMAND ----------

# MAGIC %md As part of a production-like pipeline, we externalize the column names as configs. This simplifies the work to adapt the pipeline to similar data sources in case column names are different.

# COMMAND ----------

# DBTITLE 1,Configure the column names in our tables and data source path
FILENAME = "filename"
DIRECTORY = "directory"
CONTENT = "content"
CONTENTS_CFG = "config_content"
CONTENTS_DAT = "dat_content"

# Output Columns from UDF
TIME = "timestamp"
TIME_MILLIS = "time_millis"
TIME_MICRO = "microseconds"
VALUE = "value"
CHANNEL_ID = "channel_id"
CHANNEL_TYPE = "channel_type"
ANALOG = "analog"
ANALOG_UNITS = "analog_units"
STATUS = "status"
STATION_NAME = "station_name"
REC_DEV_ID = "rec_dev_id"
REV_YEAR = "rev_year"
FREQ = "frequency"
PHASE = "phase"
ANALOG_CHANNEL_NAMES = "analog_channel_names"
STATUS_CHANNEL_NAMES = "status_channel_names"

## Other configs
# This is the path where new COMTRADE files (.cfg, .dat) stream in. Here we reuse the same source data for model training for illustration. In realistic deployments this would be a diffent path.
COMTRADE_DELTA_LAKE_PATH = "s3://db-gtm-industry-solutions/data/rcg/comtrade/source"

# model name in model registry
model_name = "fault_detection"

# COMMAND ----------

# MAGIC %md
# MAGIC # Bronze : Raw File Extractions
# MAGIC 
# MAGIC The logic of the bronze layer tables below are elaborated in Notebook 02.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze : Config Files

# COMMAND ----------

CONFIG_FILES_BRONZE_TABLE = "config_files_bronze"

@dlt.table(
    name=CONFIG_FILES_BRONZE_TABLE,
    comment="Raw .cfg files",
    table_properties={"quality" : "bronze"}
)
def config_files_bronze():
    return (
        spark
        .readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryfile")
        .option("pathGlobFilter", "*.cfg") # we look for all files in .cfg format ignoring the folder structure they come in
        .load(COMTRADE_DELTA_LAKE_PATH)
        .withColumn(FILENAME, F.element_at(F.split(F.input_file_name(),"\."),1))
        .withColumn(CONTENTS_CFG, F.col(CONTENT).cast("string"))
        .drop("path",CONTENT,"length")
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze : Data Files

# COMMAND ----------

DAT_FILES_BRONZE_TABLE = "dat_files_bronze"

@dlt.table(
    name=DAT_FILES_BRONZE_TABLE,
    comment="Raw .dat files",
    table_properties={"quality" : "bronze"}
)
def dat_files_bronze():
    return (
        spark
        .readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "binaryfile")
        .option("pathGlobFilter", "*.dat") # we look for all files in .dat format ignoring the folder structure they come in
        .load(COMTRADE_DELTA_LAKE_PATH)
        .withColumn(FILENAME, F.element_at(F.split(F.input_file_name(),"\."),1))
        .withColumnRenamed(CONTENT,CONTENTS_DAT) # Rename the content column
        .drop("path","length") # Drop unneeded columns
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze : Joined Config and Data Files

# COMMAND ----------

JOINED_FILES_BRONZE_TABLE = "joined_files_bronze"

@dlt.table(
    name=JOINED_FILES_BRONZE_TABLE,
    comment="Joined .cfg and .dat files on filename",
    table_properties={"quality" : "bronze"}
)
def joined_files_bronze():
    cfg_stream = dlt.read_stream(CONFIG_FILES_BRONZE_TABLE).withColumnRenamed(FILENAME, "cfg_filename").withColumnRenamed("modificationTime","cfg_mod_time")
    dat_stream = dlt.read_stream(DAT_FILES_BRONZE_TABLE).withColumnRenamed(FILENAME, "dat_filename").withColumnRenamed("modificationTime","dat_mod_time")
    return (
        cfg_stream
        .join(
            dat_stream,
            F.expr("""
            cfg_filename = dat_filename AND
            abs(cfg_mod_time - dat_mod_time) <= interval 5 minutes
            """),
            how = "inner"
        )
        .select(
            F.col("cfg_filename").alias(FILENAME),
            F.col("config_content"),
            F.col("dat_content")
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver : Comtrade Processing
# MAGIC 
# MAGIC The logic of the silver tables below are explained in detail in notebook 02.

# COMMAND ----------

# DBTITLE 1,Define Function to Convert COMTRADE Data to JSON
@fn.udf('string')
def get_comtrade_as_json(cfg_content: bytes, dat_content: bytes):

  # initialize comtrade object
  ct = Comtrade()

  # read configuration into comtrade object
  ct._cfg.read( cfg_content.decode() )
  ct._cfg_extract_channels_ids(ct._cfg)
  ct._cfg_extract_phases(ct._cfg)
  
  # read data into comtrade object
  dat = ct._get_dat_reader()
  dat.read(
    dat_content.decode() if (ct.ft == "ASCII") else dat_content, # determine whether data expected as binary or text
    ct._cfg
    )
  ct._dat_extract_data(dat)

  # initialize ct dictionary object
  ct_dict = {}

  # get config values
  ct_dict[FREQ] = ct.frequency
  ct_dict[REC_DEV_ID] = ct.rec_dev_id
  ct_dict[STATION_NAME] = ct.station_name
  ct_dict[TIME_MICRO] = [int(ct.start_timestamp.timestamp()) + int(second * 1e6) for second in ct.time]

  # process analog channel info
  if ct.analog_count > 0:

    # read analog, concatonate values along first access, transpose axes and convert to list
    ct_dict[ANALOG] = np.vstack(ct.analog).transpose().tolist()

    # read analog units
    ct_dict[ANALOG_UNITS] = [c.uu for c in ct._cfg.analog_channels]

    # read analog channel names
    ct_dict[ANALOG_CHANNEL_NAMES] = ct.analog_channel_ids

  # process status info
  if ct.status_count > 0:
    ct_dict[STATUS] = np.vstack(ct.status).transpose().tolist()
    ct_dict[STATUS_CHANNEL_NAMES] = ct.status_channel_ids

  # return dictionary object as json string
  return json.dumps(ct_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output JSON Schema

# COMMAND ----------

json_schema = StructType([
    StructField(ANALOG, ArrayType(ArrayType(DoubleType()))),
    StructField(ANALOG_UNITS, ArrayType(StringType())),
    StructField(STATUS, ArrayType(ArrayType(DoubleType()))),
    StructField(ANALOG_CHANNEL_NAMES, ArrayType(StringType())),
    StructField(STATUS_CHANNEL_NAMES, ArrayType(StringType())),
    StructField(FREQ, DoubleType()),
    StructField(REC_DEV_ID, StringType()),
    StructField(STATION_NAME, StringType()),
    StructField(TIME, ArrayType(LongType()))
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver : Comtrade Processing

# COMMAND ----------

COMTRADE_SILVER_TABLE = "comtrade_json_silver"

@dlt.table(
    name=COMTRADE_SILVER_TABLE,
    comment="Processed Comtrade JSON",
    table_properties={"quality" : "silver"}
)
def comtrade_json_silver():
    return (
        dlt
        .read_stream(JOINED_FILES_BRONZE_TABLE)
        .withColumn("string_json", get_comtrade_as_json(F.col("config_content"), F.col("dat_content")))
        .withColumn("parsed_json", F.from_json(F.col("string_json"), json_schema))
        .select("*", "parsed_json.*")
        .drop("parsed_json")
        .withColumn(TIME, F.transform(TIME, lambda x : F.to_timestamp(x / 1e6)))
        .withColumn("processed_timestamp", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver : Flattened data
# MAGIC This flattened table is an optional transformation for the ease of interactive queries. We explode the array and flatten the struct columns for users to interact with the data more easily.

# COMMAND ----------

FLATTENED_DATA_SILVER = "flattened_silver"

@dlt.table(
    name=FLATTENED_DATA_SILVER,
    comment="Flattened COMTRADE data",
    table_properties={"quality" : "silver"}
)
def flattened_data_silver():
    array_columns_to_bundle = [TIME, "analog"] # Include status here too, if you need status channels
    unneeded_columns = ["analog_units","status", "status_channel_names", "config_content", "dat_content", "binary_json"]
    return (
        dlt
        .read_stream(COMTRADE_SILVER_TABLE)
        .drop(*metadata_cols, *unneeded_columns)
        .select(
            "*",
            F.arrays_zip(*array_columns_to_bundle).alias("array_cols")
        )
        .drop(*array_columns_to_bundle)
        .select("*",F.explode("array_cols").alias("struct_col"))
        .drop("array_cols")
        .select("*","struct_col.*")
        .drop("struct_col")
        .withColumn("analog_channels_per_timestamp", F.arrays_zip("analog_channel_names","analog"))
        .drop("analog_channel_names","analog")
        .select("*", F.explode("analog_channels_per_timestamp").alias("analog_channel"))
        .drop("analog_channels_per_timestamp")
        .select("*", "analog_channel.*")
        .drop("analog_channel")
    )


# COMMAND ----------

# MAGIC %md 
# MAGIC # Silver : Metadata

# COMMAND ----------

METADATA_SILVER_TABLE = "metadata_silver"
metadata_cols = [FREQ, STATION_NAME, REC_DEV_ID]

@dlt.table(
    name=METADATA_SILVER_TABLE,
    comment="COMTRADE Extracted Record Metadata",
    table_properties={"quality" : "silver"}
)
def comtrade_metadata_silver():
    return (
        dlt
        .read_stream(COMTRADE_SILVER_TABLE)
        .withWatermark("processed_timestamp", "10 seconds")
        .select(FILENAME, *metadata_cols)
        .groupby(FILENAME)
        .agg(
            *[F.first(metadata_col).alias(metadata_col) for metadata_col in metadata_cols]
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Silver : Create columns for current Channels

# COMMAND ----------

# MAGIC %md
# MAGIC In this step, we create separate columns for `IA`, `IB` and `IC` channels similar to how one *pivots* a table. 
# MAGIC 
# MAGIC As of today <a href = "https://docs.databricks.com/workflows/delta-live-tables/delta-live-tables-python-ref.html?_ga=2.22595967.59249201.1678131937-1347012581.1661879485#limitations">`.pivot` is not supported in DLT</a>, as DLT does not support any operations where table values impacts the schema of subsequent tables. However, in our case, because there are a set number of potential channels (3), and the data for each COMTRADE is small enough, we can define a pandas_udf that is compatible with DLT to circumvent this problem.

# COMMAND ----------

# DBTITLE 1,Schema Defintion for the UDF
pivoted_schema = StructType([
        StructField(FILENAME, StringType(), False),
        StructField(TIME, TimestampType(), False),
        StructField("IA", DoubleType(), True),
        StructField("IB", DoubleType(), True),
        StructField("IC", DoubleType(), True),
        StructField("processed_timestamp", TimestampType(), False)
    ])

# COMMAND ----------

# DBTITLE 1,UDF for "pivoting"
def pivot_current_channels(df : pd.DataFrame) -> pd.DataFrame:
    _processed_timestamp = df["processed_timestamp"].iloc[0]
    _pivoted = df.pivot_table(index=[FILENAME,TIME], columns="analog_channel_names", values="analog", aggfunc="first").reset_index(drop=False)
    _pivoted["processed_timestamp"] = _processed_timestamp
    return _pivoted

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver : Pivot Current Channels

# COMMAND ----------

PIVOT_CURRENT_SILVER = "pivoted_current_silver"
current_channels = ["IA", "IB", "IC"]

@dlt.table(
    name=PIVOT_CURRENT_SILVER,
    comment="Pivoted current channels (IA, IB, IC)",
    table_properties={"quality" : "silver"}
)
def pivoted_current_silver():
    return (
        dlt
        .read_stream(FLATTENED_DATA_SILVER)
        .withWatermark("processed_timestamp", "1 second")
        .filter(F.col("analog_channel_names").isin(*current_channels))
        .groupby(FILENAME)
        .applyInPandas(pivot_current_channels, pivoted_schema)
    )

# COMMAND ----------

# MAGIC %md
# MAGIC # Gold : Electrical Fault Detection

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference function
# MAGIC 
# MAGIC From the registry, we can retrieve the production version of our fault detection model and create an inference function with the model.
# MAGIC 
# MAGIC We broadcast the model and load the model from the broadcasted variable in the pandas UDF. This technique further improves the efficiency of loading a model from the registry: the model is loaded from mlflow model registry only once and then copied to cluster workers via a broadcast variable.
# MAGIC 
# MAGIC We use an a Pandas UDF with Iterator support for our inference function to further reduce the number of times we need to load a model from registry. This pandas UDF type is useful when the UDF execution requires initializing some state, for example, loading a machine learning model file to apply inference to every input batch. 

# COMMAND ----------

# Load the fault model from mlflow and broadcast to worker nodes
fault_bc = spark.sparkContext.broadcast(ml_keras.load_model(f"models:/{model_name}/Production"))

@pandas_udf(DoubleType())
def fault_identification(waveform_series : Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Load model
    fault_model = fault_bc.value
    for waveform_pd in waveform_series:
        wfs = np.vstack(waveform_pd.map(lambda x: np.expand_dims(np.vstack(x),0)).tolist())
        scores = np.squeeze(fault_model.predict(wfs)).tolist()
        yield pd.Series(scores)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold : Electrical Fault Detection

# COMMAND ----------

FAULT_DETECTION_GOLD = "fault_detection_gold"

@dlt.table(
    name=FAULT_DETECTION_GOLD,
    comment="waveform 3-phase fault detection using Tensorflow model",
    table_properties={"quality" : "gold"}
)
def electrical_fault_detection_gold():
    return (
        dlt
        .read_stream(PIVOT_CURRENT_SILVER)
        .withWatermark("processed_timestamp", "1 second")
        .select(F.struct(F.col("timestamp"),*[F.col(cur_chan).alias(cur_chan) for cur_chan in current_channels]).alias("timestep"),"filename","processed_timestamp")
        .groupby(FILENAME)
        .agg(
            F.collect_list(F.col("timestep")).alias("timestep"),
            F.first("processed_timestamp").alias("processed_timestamp")
        )
        .withColumn("timestep", F.array_sort("timestep"))
        .withColumn("timestep_array", F.transform("timestep", lambda x: F.array(*[x[cur_chan] for cur_chan in current_channels])))
        .withColumn("fault_score", fault_identification(F.col("timestep_array")))
        .select(FILENAME, "processed_timestamp","fault_score")
    )

# COMMAND ----------

# MAGIC %md # Deploying the whole pipeline

# COMMAND ----------

# DBTITLE 1,DLT Deployment
# MAGIC %md We recommemnd you use the RUNME notebook to automate the creation of the Workflow for this accelerator - the DLT pipeline is the last step in the automated Workflow.
# MAGIC 
# MAGIC Alternatively, we can configure the DLT pipeline using the UI according to the screenshots below. 
# MAGIC 
# MAGIC * In the *Create Pipeline* dialog, we select *04_Fault_Detection_DLT*.
# MAGIC 
# MAGIC * Under *Target*, we specify the name of the database within which DLT objects created in these workflows should reside. Enter `solacc_fault_detection`.
# MAGIC 
# MAGIC * Under *Storage Location*, we specify the storage location where object data and metadata of the DLT will be placed. For a matching example to the automated DLT, enter `/databricks_solacc/fault_detection/dlt`.
# MAGIC 
# MAGIC Under *Pipeline Mode*, we specify how the cluster that runs our job will be managed.  If we select *Triggered*, the cluster shuts down with each cycle.  As several of our DLT objects are configured to run continously, we should select *Continous* mode. In our DLT object definitions, we leveraged some throttling techniques to ensure our workflows do not become overwhelmed with data.  Still, there will be some variability in terms of data moving through our pipelines so we might specify a minimum and maximum number of workers within a reasonable range based on our expectations for the data.  Once deployed, we might monitor resource utilization to determine if this range should be adjusted.
# MAGIC 
# MAGIC **NOTE** Continous jobs will run indefinetly until explicitly stopped.  Please be aware of this as you manage your DLT pipelines.
# MAGIC 
# MAGIC Clicking *Create* we now have defined the jobs for our DLT workflow. You can compare the `Settings` of the created pipeline with the standard settings. Below are the standard settings our RUNME automation notebook uses:
# MAGIC 
# MAGIC <img src='https://github.com/databricks-industry-solutions/comtrade-accelerator/raw/main/images/dlt-config2.png' width=800>
# MAGIC 
# MAGIC <img src='https://github.com/databricks-industry-solutions/comtrade-accelerator/raw/main/images/dlt-config.png' width=800>

# COMMAND ----------

# DBTITLE 1,DLT Monitoring
# MAGIC %md
# MAGIC After running the overall Workflow created by the RUNME notebook, or after running the individual notebooks interactively in order and then running the DLT pipeline, you should see the following 
# MAGIC 
# MAGIC <img src='https://github.com/databricks-industry-solutions/comtrade-accelerator/raw/main/images/dlt.png' width=800>
# MAGIC 
# MAGIC Each box represents a table we define with `@dlt.table` in this notebook. Each table contains stats related to execution duration, record count, and optionally data quality information.
# MAGIC 
# MAGIC The connections between the items indicate the dependencies between objects.  Color coding indicates the status of the tables in the pipeline. Should an error be encountered, event information at the bottom of the UI would reflect this.  Clicking on the error event would then expose error messages with which the problem could be diagnosed.
# MAGIC 
# MAGIC When the job initialy runs, it will run in *Development* mode as indicated at the top of the UI.  In Development mode, any errors will cause the job to be stopped so that they may be corrected. By clicking *Production*, the job is moved into a state where jobs are restarted upon error.

# COMMAND ----------



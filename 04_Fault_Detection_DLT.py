# Databricks notebook source
# MAGIC %pip install comtrade orjson mlflow tensorflow

# COMMAND ----------

# MAGIC %md In this notebook, we combine all the ETL steps from notebook 02 with the ML model trained in notebook 03 to build a DLT pipeline for end-to-end model inference. 

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

# This is the path where new COMTRADE files (.cfg, .dat) stream in. Here we reuse the same source data for model training for illustration. In realistic deployments this would be a diffent path.
COMTRADE_DELTA_LAKE_PATH = "s3://db-gtm-industry-solutions/data/rcg/comtrade/source"

# COMMAND ----------

# Load the fault model from mlflow and broadcast to worker nodes
fault_bc = spark.sparkContext.broadcast(ml_keras.load_model("models:/fault_detection/Production"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Bronze : Raw File Extractions

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

def read_comtrade_dynamic(cfg : str, dat : bytes) -> comtrade.Comtrade:
    # NEW : Create new instance of comtrade.Comtrade
    ct = comtrade.Comtrade()
    # These lines are the same as Comtrade.read():
    ct._cfg.read(cfg)
    ct._cfg_extract_channels_ids(ct._cfg)
    ct._cfg_extract_phases(ct._cfg)
    dat_proc = ct._get_dat_reader()
    # NEW : Add the following line to dynamically determine if the bytes object should be converted to a string with .decode()
    dat = dat.decode() if (ct.ft == "ASCII") else dat
    # Below lines the same as Comtrade.read()
    dat_proc.read(dat,ct._cfg)
    ct._dat_extract_data(dat_proc)
    # NEW : return the comtrade.Comtrade object.
    return ct

def retrieve_dict(cfg : str, dat : bytes) -> str:
    # Pass in the config file string and data file binary contents to create a comtrade.Comtrade object
    _comtrade = read_comtrade_dynamic(cfg, dat)
    # How many analog and status channels exist?
    analog_count = _comtrade.analog_count
    status_count = _comtrade.status_count
    # initialize an empty dicitonary
    ret = {}
    # Initialize empty analog return values
    _analog_list = []
    _analog_units = []
    _analog_channel_names = []
    if (analog_count > 0):
        # Use numpy.vstack to stack the analog channels into an np.array, transpose it, then convert it to a List[List[float]] object. 
        # where each list corresponds to the value of all analog channels at a specify timestamp.
        _analog_list = np.vstack(_comtrade.analog).transpose().tolist()
        _analog_channels = [channel.uu for channel in _comtrade._cfg.analog_channels]
        _analog_channel_names = _comtrade.analog_channel_ids
    ret[ANALOG] = _analog_list
    ret[ANALOG_UNITS] = _analog_units
    ret[ANALOG_CHANNEL_NAMES] = _analog_channel_names
    # Initialize empty status return values
    _status_list = []
    _status_channel_names = []
    if (status_count > 0):
        # get the value of each status channel at each timestamp
        _status_list = np.vstack(_comtrade.status).transpose().tolist()
        _status_channel_names = _comtrade.status_channel_ids
    ret[STATUS] = _status_list
    # get the frequency, rec_dev_id, and station name
    ret[FREQ] = _comtrade.frequency
    ret[REC_DEV_ID] = _comtrade.rec_dev_id
    ret[STATION_NAME] = _comtrade.station_name
    # Because spark timestamps only go down to the millisecond level, we'll need to track microseconds separately.
    _datetimes = [_comtrade.start_timestamp + datetime.timedelta(seconds = fractional_second) for fractional_second in _comtrade.time]
    ret[TIME] = [int(dt.timestamp() * 1e6) for dt in _datetimes]
    # Dump the dictionary to a binary string.
    return orjson.dumps(ret)

@pandas_udf(BinaryType())
def get_comtrade_json(cfg_series : pd.Series, dat_series : pd.Series) -> pd.Series:
    # Put the two pandas series into a pandas DataFrame
    _df = pd.DataFrame({"cfg" : cfg_series, "dat" : dat_series})
    # For every row, apply the retrieve_dict function to get the binary json string
    return _df.apply(lambda row : retrieve_dict(row["cfg"], row["dat"]),1)

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
        .withColumn("binary_json", get_comtrade_json(F.col("config_content"), F.col("dat_content")))
        .withColumn("parsed_json", F.from_json(F.col("binary_json").cast("string"), json_schema))
        .select("*", "parsed_json.*")
        .drop("parsed_json")
        .withColumn(TIME, F.transform(TIME, lambda x : F.to_timestamp(x / 1e6)))
        .withColumn("processed_timestamp", F.current_timestamp())
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
# MAGIC # Silver : Flattened Data

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
# MAGIC # Silver : Pivot Current Channels

# COMMAND ----------

# MAGIC %md
# MAGIC As of today <a href = "https://docs.databricks.com/workflows/delta-live-tables/delta-live-tables-python-ref.html?_ga=2.22595967.59249201.1678131937-1347012581.1661879485#limitations">`.pivot` is not supported in DLT</a>. Since the data for each COMTRADE is small enough, below will be defined a pandas_udf which can pivot the data. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema Defintion

# COMMAND ----------

pivoted_schema = StructType([
        StructField(FILENAME, StringType(), False),
        StructField(TIME, TimestampType(), False),
        StructField("IA", DoubleType(), True),
        StructField("IB", DoubleType(), True),
        StructField("IC", DoubleType(), True),
        StructField("processed_timestamp", TimestampType(), False)
    ])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helper Functions

# COMMAND ----------

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
# MAGIC ## Helper Functions

# COMMAND ----------

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

# Databricks notebook source
# MAGIC %md The purpose of this notebook is to read the COMTRADE data files as part of the COMTRADE solution accelerator.  This notebook was developed on a **Databricks ML 12.2 LTS** cluster. This notebook is also available at https://github.com/databricks-industry-solutions/comtrade-accelerator.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC We now have a dataset available in the COMTRADE format.  In this format, data is received in two files.  The configuration file (CFG) contains the higher-level details about a given set of data while the data file (DAT) contains a series of readings. These file are often read together using format aware libraries, making the large scale processing of these data using general purpose engines such as Spark a bit tricky.  Furthermore, the structure of the data read from these files isn't often aligned with the needs of general purpose analytics tools so that we often need to restructure the data to make it more widely accessible. In this notebook, we'll tackle both these challenges in preparation for the analytics work to take place in downstream notebooks.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install comtrade 

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn
from pyspark.sql.types import *

from comtrade import Comtrade

import numpy as np
import json

import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Read the COMTRADE Data
# MAGIC 
# MAGIC With the COMTRADE data delivered as two files, *i.e.* a CFG and a DAT file, the easiest way to process the incoming data is to read each file separately and combine them based on matching file name (ignoring the respective *.cfg* and *.dat* file extensions). To do this, we'll reach each file as a single entity using the [*binaryFile*](https://spark.apache.org/docs/latest/sql-data-sources-binaryFile.html) format.  This will create one record for each file with the following fields:
# MAGIC </p>
# MAGIC 
# MAGIC * path: a string representing the full name of the file
# MAGIC * modificationTime: a timestamp value representing the date teh file was last modified
# MAGIC * length: a long integer value representing the number of bytes associated with the file
# MAGIC * content: a binary value representing the file's contents
# MAGIC 
# MAGIC For each file type, we will parse the file name minus the file extension  to enable a match between the CFG and the DAT files.  Please note that each file resides in a irregular folder hierarchy where as Spark typically prefers to read files within a single folder.  We'll enable a recursive read to allow us to read from across the entire folder substructure:

# COMMAND ----------

# DBTITLE 1,Read CFG Files
cfg = (
  spark 
    .read
    .format('binaryFile')
    .option('recursiveFileLookup','true') # enable reading from all subfolders
    .option('pathGlobFilter', '*.cfg') # .cfg files only
    .load(config['source_path']) 
    .withColumn('base_name', fn.expr("split(path, '[.]')[0]"))
  )

display(cfg.limit(10))

# COMMAND ----------

# DBTITLE 1,Read DAT Files
dat = (
  spark 
    .read
    .format('binaryFile')
    .option('recursiveFileLookup','true') # enable reading from all subfolders
    .option('pathGlobFilter', '*.dat') # .dat files only
    .load(config['source_path']) 
    .withColumn('base_name', fn.expr("split(path, '[.]')[0]"))
  )

display(dat.limit(10))

# COMMAND ----------

# MAGIC %md We can now link these two datasets together in preparation for further transformations:

# COMMAND ----------

# DBTITLE 1,Merge Datasets
ctrade_raw = (
  cfg.alias('cfg')
    .join(
      dat.alias('dat'), 
      on='base_name', 
      how='inner'
      )
    .selectExpr(
        'cfg.path as path_cfg',
        'dat.path as path_dat',
        'cfg.content as content_cfg',
        'dat.content as content_dat'
      )
  ).cache() 

display( ctrade_raw.limit(10) )

# COMMAND ----------

# MAGIC %md ##Step 2: Restructure the Data for Analysis
# MAGIC 
# MAGIC With the raw data now assembled, we need to interpret the content from the CFG-DAT file pairs as COMTRADE data and return it as a more accessible data object.  Because of the relationship between the configuration and data records, a JSON structure would seem to provide a nice option for returning the data: 

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
  ct_dict['frequency'] = ct.frequency
  ct_dict['rec_dev_id'] = ct.rec_dev_id
  ct_dict['station_name'] = ct.station_name
  ct_dict['microseconds'] = [int(ct.start_timestamp.timestamp()) + int(second * 1e6) for second in ct.time]

  # process analog channel info
  if ct.analog_count > 0:

    # read analog, concatonate values along first access, transpose axes and convert to list
    ct_dict['analog'] = np.vstack(ct.analog).transpose().tolist()

    # read analog units
    ct_dict['analog_units'] = [c.uu for c in ct._cfg.analog_channels]

    # read analog channel names
    ct_dict['analog_channel_names'] = ct.analog_channel_ids

  # process status info
  if ct.status_count > 0:
    ct_dict['status'] = np.vstack(ct.status).transpose().tolist()
    ct_dict['status_channel_names'] = ct.status_channel_ids

  # return dictionary object as json string
  return json.dumps(ct_dict)

# COMMAND ----------

# MAGIC %md There's a lot going on in the function above.  You can read more about the individual comtrade object methods and attributes in the [comtrade library's online documentation](https://github.com/dparrini/python-comtrade), though the documentation is a bit sparse.  The key thing to take from this is that we will be returned a JSON string with the following keys: 
# MAGIC </p>
# MAGIC 
# MAGIC * station_name - name of the substation location
# MAGIC * rec_dev_id - identification number or name of the device
# MAGIC * frequency -  nominal line frequency in Hz
# MAGIC * microseconds - the microseconds timestamp associated with the analog readings
# MAGIC * analog - analog readings as a list of lists of floating point values
# MAGIC * analog_units - units associated with analog readings, e.g. amperes
# MAGIC * analog_channel_names - the names of the channels for which readings are recorded
# MAGIC * status - status readings for the different channels
# MAGIC * status_channel_names - the names of the status channels
# MAGIC 
# MAGIC **NOTE** See Appendix A in [this document](https://ckm-content.se.com/ckmContent/sfc/servlet.shepherd/document/download/0691H00000GYrXeQAL) for additional information about these and other fields in the COMTRADE format specification.

# COMMAND ----------

# DBTITLE 1,Convert COMTRADE Data to JSON
# structure of the JSON we expect to be returned by the function
ctrade_schema = StructType([
  StructField('station_name', StringType()),
  StructField('rec_dev_id', StringType()),
  StructField('frequency', FloatType()),
  StructField('microseconds', ArrayType(LongType())),
  StructField('analog', ArrayType(ArrayType(DoubleType()))),
  StructField('analog_units', ArrayType(StringType())),
  StructField('analog_channel_names', ArrayType(StringType())),
  StructField('status', ArrayType(ArrayType(StringType()))),
  StructField('status_channel_names', ArrayType(StringType()))
  ])

# convert comtrade to json and then json to structure
ctrade = (
  ctrade_raw
    .withColumn('ctrade',get_comtrade_as_json('content_cfg','content_dat'))
    .withColumn('ctrade',fn.from_json('ctrade', ctrade_schema))
    .drop('content_cfg','content_dat')
    .select('path_cfg','path_dat','ctrade')
  )

display(ctrade.limit(10))

# COMMAND ----------

# MAGIC %md While we have our data parsed, there are many elements we might want to access. The metadata for the files is the easiest to extract with one row per COMTRADE file pair:

# COMMAND ----------

# DBTITLE 1,Metadata
# extract header metadata
metadata = (
  ctrade
    .selectExpr(
      'path_cfg',
      'path_dat',
      'ctrade.station_name as station_name',
      'ctrade.rec_dev_id as rec_dev_id', 
      'ctrade.frequency as frequency'
      )
)

# write results to table
_ = (
  metadata
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('metadata')
  )

display(spark.table('metadata'))

# COMMAND ----------

# MAGIC %md For readings, it the data manipulations are a bit more complex.  In the current state of the dataset, we have the microseconds time value for each reading in its own array.  The channel readings are held as an array of arrays where the outer array is aligned with the microseconds values and the inner array represents the readings for each channel at that point in time.  What we need to do is link the microseconds time values with the readings and get each channel reading into its own field.  This will require us to [zip together arrays](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.arrays_zip.html#pyspark.sql.functions.arrays_zip) and [explode](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.explode.html#pyspark.sql.functions.explode) our data so that individual entries in the arrays are moved to rows within the resulting dataset.  We will need to do this a few times and then eventually [pivot](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.pivot.html#pyspark.sql.GroupedData.pivot) our data until we arrive at the final result set shown here:

# COMMAND ----------

# DBTITLE 1,Readings
# extract readings data
readings = (
  ctrade
    .selectExpr(
      'path_cfg',
      'path_dat',
      'ctrade.analog_channel_names as analog_channel_names',
      ''' 
       arrays_zip(
        ctrade.microseconds,
        ctrade.analog
        ) as reading
        ''' # produces array of {microseconds: ###, analog:[channel 0, channel 1, ... channel n]}
      )
    .withColumn('reading', fn.explode('reading')) # splits each reading in reading array to a row in resulting table
    .withColumn('microseconds', fn.col('reading.microseconds')) # get microseconds value to its own field
    .withColumn('reading', fn.arrays_zip('analog_channel_names', 'reading.analog')) # combine channel names with readings [{analog channel name, analog value}]
    .drop('analog_channel_names')
    .withColumn('reading', fn.explode('reading')) # move each channel reading to its own row
    .withColumn('analog_channel_name', fn.col('reading.analog_channel_names')) # move channel name to its own field
    .withColumn('analog', fn.col('reading.analog')) # move analog value to its own field
    .drop('reading')
    .groupBy('path_cfg','path_dat','microseconds') # group by comtrade file id and microseconds
      .pivot('analog_channel_name')  # pivot analog channel name to columns
      .agg(fn.first('analog'))  # move first value (and only) value for that column for this id + microseconds to be column value
    )

_ = (
  readings
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('readings')
  )


display(spark.table('readings'))

# COMMAND ----------

# MAGIC %md **NOTE** If we were incrementally updating a table with newly arrived COMTRADE data, we might employ a [merge](https://docs.databricks.com/delta/merge.html#language-python) instead of the table overwrite logic demonstrated here.  In that merge, the *path_cfg* and the *path_dat* fields would be used as the basis for joining records between the source and target tables.  In such a scenario, it is important to take advantage of the [schema evolution](https://docs.databricks.com/delta/update-schema.html#add-columns-with-automatic-schema-update) capabilities of the Delta Lake format to accommodate variations in file structures that may be observed.

# COMMAND ----------

# MAGIC %md As we did in the last notebook, we might plot one of the COMTRADE files to verify its data has been captured appropriately. The COMTRADE file we use as example here is the same one we downloaded in Notebook 01.

# COMMAND ----------

# DBTITLE 1,Plot COMTRADE File
# get name of one output CFG file
sample_output_file_name = "/databricks/driver/cap1f_01.cfg"

# instantiate comtrade format reader
comtrade_data = Comtrade()

# loade the CFG and associated DAT file
comtrade_data.load(
  sample_output_file_name,
  sample_output_file_name.replace('.cfg','.dat')
)

# plot the readings from the file
plt.figure(figsize=(10,8))
plt.plot(comtrade_data.analog[0])
plt.plot(comtrade_data.analog[1])
plt.plot(comtrade_data.analog[2])
plt.title(sample_output_file_name)
plt.show()

# COMMAND ----------

# DBTITLE 1,Plot Table Data
# get readings data from spark
sample_data = (
  spark
    .table('readings')
    .filter("path_cfg like '%cap1f_01.cfg%'")
    .orderBy('microseconds')
    .select('IA','IB','IC')
  )
# send to pandas dataframe
sample_data_pd = sample_data.toPandas()

# plot the readings from the dataframe
plt.figure(figsize=(10,8))
plt.plot(sample_data_pd['IA'])
plt.plot(sample_data_pd['IB'])
plt.plot(sample_data_pd['IC'])
plt.title(sample_output_file_name)
plt.show()

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | comtrade | A module designed to read Common Format for Transient Data Exchange (COMTRADE) file format |  MIT | https://pypi.org/project/comtrade/                       |
# MAGIC | comtradehandlers | File handlers for the COMTRADE format| MIT | https://github.com/relihanl/comtradehandlers.git#egg=comtradehandlers |

# Databricks notebook source
# MAGIC %md The purpose of this notebook is to prepare the data for the COMTRADE solution accelerator.  This notebook was developed on a **Databricks ML 12.2 LTS** cluster. This notebook is also available at https://github.com/databricks-industry-solutions/comtrade-accelerator.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC In this notebook, we'll take a set of text files representing simulated current readings in a *5-bus interconnected system for Phase Angle Regulators and Power Transformers* and convert them to the [COMTRADE format](https://ieeexplore.ieee.org/document/6512503) as defined by the [Institute of Electrical and Electronic Engineers (IEEE)](https://www.ieee.org/). This format is widely used in a number of analytic applications, making it critical that we demonstrate how such data can be processed in Databricks.
# MAGIC 
# MAGIC The dataset we will be using is the IEEE's [*Transients and Faults in Power Transformers and Phase Angle Regulators* dataset](https://ieee-dataport.org/open-access/transients-and-faults-power-transformers-and-phase-angle-regulators-dataset).  The files that makeup this dataset are generated using EMTDC/PSCAD and are provided in a simple, four-column delimited text format, hence the need for this initial conversion.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install comtrade git+https://github.com/relihanl/comtradehandlers.git#egg=comtradehandlers

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

from comtradehandlers import writer
from comtrade import Comtrade

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from datetime import datetime

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Access Data Files
# MAGIC 
# MAGIC  should be downloaded, unzipped and uploaded to a cloud storage account accessible to your Databricks workspace.  This storage account should then be mounted to your workspace using [these instructions](https://docs.databricks.com/dbfs/mounts.html). For the purposes of this notebook, we've uploaded these files to a mount point named */mnt/comtrade* and placed the files in a folder named *input*.  You can configure these settings in notebook *00* if you require different paths.
# MAGIC 
# MAGIC The data files provided with the dataset are organized in a fairly ragged folder hierarchy which you can see here:

# COMMAND ----------

# DBTITLE 1,Copy source data into input path - this can take more than an hour
# dbutils.fs.cp(config['source_path'], config['input_path'], True)

# COMMAND ----------

# DBTITLE 1,List Files in Input Path
# define function to enumerate folder structure
def list_folder_contents(path, level=0):

  # initialize variables
  i = 0
  subfolders = []
  this_indent = "--" * level

  # if first folder, get full path
  if level==0:
    this_folder=path
  else: # otherwise, get folder name
    this_folder = path.split('/')[-2]

  # for each file in this folder
  for file in dbutils.fs.ls(path):
    
    # if directory, capture path 
    if file.size == 0:
      subfolders += [file.path]
    else: # if file, count it
      i += 1

  # print details about this folder (folder icon | folder name (file count)
  print(this_indent, "\U0001F4C1", this_folder, f"({i} files)")     

  # process subfolders
  for subfolder in subfolders:
    i += list_folder_contents(subfolder, level+1)

  return i

# capture folder structure details
file_count = list_folder_contents(config['input_path'])
print(f"{file_count} total files found")

# COMMAND ----------

# MAGIC %md To access these files, we'll simply compile a list of the individual files as follows:

# COMMAND ----------

# DBTITLE 1,Get List of Input Files
# define function to get list of files
def get_file_names(path):

  # initialize files list
  file_names = []

  # for each item in folder
  for obj in dbutils.fs.ls(path):
 
    # if subfolder
    if obj.size==0:
      file_names += get_file_names(obj.path) # get files in subfolder

    # if file
    else: 
      file_names += [[obj.path]] # file name is treated as the value in a single-item list
    

  return file_names

# get list of input file names
input_file_names = get_file_names(config['input_path'])

# COMMAND ----------

# MAGIC %md And we can review the structure of one of the files to get a sense of its structure:

# COMMAND ----------

# DBTITLE 1,Display Contents of One File
print(
  dbutils.fs.head(input_file_names[0][0])
  )

# COMMAND ----------

# MAGIC %md Each file in this dataset, which we will refer to as our input files, adheres to this structure.  Each record represents a three-phase reading taken at a particular point in time. The first field represents the *time* increment from the start of the simulation and the subsequent fields represent the current associated with phases *A*, *B* and *C*, measures in amperes.  These details will be important as we convert each of these files to the COMTRADE format.

# COMMAND ----------

# MAGIC %md ##Step 2: Convert Files to COMTRADE Format
# MAGIC 
# MAGIC To convert each file to the COMTRADE format, we'll define a simple function.  This function will receive the path of the input text file along with values for simulated parameters required for use with the COMTRADE format. With each call, the function will open up a ComtradeWriter object, read the data from the input file, and assign it to the COMTRADE output.  The result will be one COMTRADE configuration \(CFG\) file and one data file \(DAT\) for each input text file.  The output will be written to an output folder location, *i.e.* */mnt/comtrade/output* per our default configurations in notebook *00*, under an identical folder structure to the one within which the original input file resides:

# COMMAND ----------

# DBTITLE 1,Conversion Function
@fn.udf('string')
def write_comtrade_files(input_path: str, station_name: str, rec_dev_id: str, start_time: datetime, trigger_time: datetime):

  # determine local input path
  local_input_path = input_path.replace('dbfs:/','/dbfs/')

  # determine local output path for CFG file
  local_output_path = ( 
    '/dbfs' + config['output_path'] + '/'+  # prefix with local output path
    '/'.join(
        local_input_path.split('/')[len(('/dbfs' + config['input_path']).split('/')):] # strip local input path from name
      )
    )
  local_output_path = local_output_path.replace('.txt', '.cfg') # replace file extension

  # make sure directory for output file exists
  Path(
    '/'.join(local_output_path.split('/')[:-1]) # folder housing this file
    ).mkdir(parents=True, exist_ok=True) # make intermediate directories if needed

  # create and configure comtrade writer
  comtrade_writer = writer.ComtradeWriter( # initialize writer
    local_output_path,
    start_time,
    trigger_time,
    station_name=station_name,
    rec_dev_id=rec_dev_id
    )
  # add currents (current-phase, phase, current, units, skew, min, max, primary, secondary)
  comtrade_writer.add_analog_channel('IA','A','I', uu='A', skew=0, min=-500, max=500, primary=1, secondary=1)
  comtrade_writer.add_analog_channel('IB','B','I', uu='A', skew=0, min=-500, max=500, primary=1, secondary=1)
  comtrade_writer.add_analog_channel('IC','C','I', uu='A', skew=0, min=-500, max=500, primary=1, secondary=1)

  # read input file records 
  data_pd = pd.read_csv(local_input_path, header=None, names=['time','IA','IB','IC']) # read input file data
  data_pd['time'] = (data_pd['time'] * 1e6) # adjust time to milliseconds

  # add input records to comtrade output
  for i, row in data_pd.iterrows():
    comtrade_writer.add_sample_record( int(row['time']), [str(row['IA']), str(row['IB']), str(row['IC'])], [])

  # write output
  comtrade_writer.finalize()

  # return output file name
  return local_output_path

# COMMAND ----------

# MAGIC %md Take a moment to note the use of the @udf decorator at the top of the function definition in the last cell.  This is used to register the previously defined function as a [user-defined scalar function](https://docs.databricks.com/udf/python.html) with the Spark engine running in Databricks.  By registering our function this way, it becomes available to be applied to a Spark dataframe.  
# MAGIC 
# MAGIC A Spark dataframe allows us to work with data in a fairly structured format.  Unlike other dataframe objects, the Spark dataframe has its contents distributed across the workers within a Spark cluster.  Logic applied to the dataframe can be executed in parallel, allowing us to accelerate the time to complete large tasks, such as the conversion of 100,000+ files from one format to another. The larger we configure the cluster, the greater the degree of parallelism and the faster the work is performed (up to a limit based on the size of the dataset).
# MAGIC 
# MAGIC Note too that the function makes use of [Python type hints](https://peps.python.org/pep-0484/).  These are optional but they do help developers setting up the dataframe against which this function will be applied understand the data types of the fields expected by the function.
# MAGIC 
# MAGIC With all that in mind, let's create a Spark dataframe with the inputs required for calls to our function:

# COMMAND ----------

# DBTITLE 1,Assemble Inputs for Function Call
inputs = (
  spark
    .createDataFrame(input_file_names, schema='input_file_name string') # convert list of input file names to dataframe
    .withColumn('station_name', fn.expr("cast(cast(10000 * rand() as int) as string)")) # station name is integer between 0 and 10000
    .withColumn('rec_dev_id', fn.expr("cast(cast(1000 * rand() as int) as string)")) # device id is integer between 0 and 1000
    .withColumn('start_time', fn.expr("current_timestamp()")) 
    .withColumn('trigger_time', fn.expr("start_time + interval 20 milliseconds"))
  )

display(inputs)

# COMMAND ----------

# MAGIC %md With all of our inputs in place, we will make sure our data are relatively evenly distributed  across our cluster and call our function to perform the conversion:

# COMMAND ----------

# DBTITLE 1,Assemble Inputs for Conversion Function
# delete any outputs from prior runs
dbutils.fs.rm(config['output_path'], recurse=True)

# define logic to convert inputs to outputs
outputs = (
  inputs
    .repartition(sc.defaultParallelism * 10) # repartition data to spread it around the cluster
    .withColumn('output_file_name', write_comtrade_files('input_file_name','station_name','rec_dev_id','start_time','trigger_time'))
)

# write results to output table, forcing all files to be converted
_ = (
  outputs
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('conversion')
  )

# display results of conversion process
conversions = spark.table('conversion')
display(
  conversions
  )

# COMMAND ----------

# MAGIC %md To verify the success of our work, we can examine the contents of the output folder.  We should expect two output files for each one input file observed earlier in this notebook:

# COMMAND ----------

# DBTITLE 1,Examine Output Folder Structure
file_count = list_folder_contents(config['output_path'])
print(f"{file_count} total files found")

# COMMAND ----------

# MAGIC %md And we can display the contents of a sample file as a graph:

# COMMAND ----------

# DBTITLE 1,Plot a Sample File
# get name of one output CFG file
sample_output_file_name = conversions.limit(1).collect()[0]['output_file_name']

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

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | comtrade | A module designed to read Common Format for Transient Data Exchange (COMTRADE) file format |  MIT | https://pypi.org/project/comtrade/                       |
# MAGIC | comtradehandlers | File handlers for the COMTRADE format| MIT | https://github.com/relihanl/comtradehandlers.git#egg=comtradehandlers |

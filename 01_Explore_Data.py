# Databricks notebook source
# MAGIC %md The purpose of this notebook is to prepare the data for the COMTRADE solution accelerator.  This notebook was developed on a **Databricks ML 12.2 LTS** cluster. This notebook is also available at https://github.com/databricks-industry-solutions/comtrade-accelerator.

# COMMAND ----------

# MAGIC %pip install comtrade

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we'll take a set of text files representing simulated current readings in a *5-bus interconnected system for Phase Angle Regulators and Power Transformers* and convert them to the [COMTRADE format](https://ieeexplore.ieee.org/document/6512503) as defined by the [Institute of Electrical and Electronic Engineers (IEEE)](https://www.ieee.org/). This format is widely used in a number of analytic applications, making it critical that we demonstrate how such data can be processed in Databricks.
# MAGIC
# MAGIC The dataset we will be using is the IEEE's [*Transients and Faults in Power Transformers and Phase Angle Regulators* dataset](https://ieee-dataport.org/open-access/transients-and-faults-power-transformers-and-phase-angle-regulators-dataset).  The files that makeup this dataset are generated using EMTDC/PSCAD and are provided in a simple, four-column delimited text format.
# MAGIC Within this dataset, we focus only on the `transient disturbances` subset that is applicable for the fault detection use case.
# MAGIC
# MAGIC We converted this subset to COMTRADE format and stored it on a publicly-assessible storage account to act as the source for our data pipeline.

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Access Data Files
# MAGIC
# MAGIC The data files provided with the dataset are organized in a fairly ragged folder hierarchy which you can see below. The `external fault with CT saturation` folder contains data that represents faults while other folders contain data that represent other conditions. We will utilize the folder structure to generate our labels for the fault detection ML model in the subsequent notebook.

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
file_count = list_folder_contents(config['source_path'])
print(f"{file_count} total files found")

# COMMAND ----------

# DBTITLE 1,Let's download a pair of .dat and .cfg files so that we can inspect locally
!wget --no-check-certificate --no-proxy -O /databricks/driver/cap1f_01.cfg https://db-gtm-industry-solutions.s3.amazonaws.com/data/rcg/comtrade/source/capacitor+switching/cap1f_01.cfg 
!wget --no-check-certificate --no-proxy -O /databricks/driver/cap1f_01.dat https://db-gtm-industry-solutions.s3.amazonaws.com/data/rcg/comtrade/source/capacitor+switching/cap1f_01.dat

# COMMAND ----------

# MAGIC %md ## Understanding the COMTRADE format
# MAGIC Our COMTRADE files are separated in CFG and DAT formats. 
# MAGIC The `.cfg` file is where metadata is saved and the `.dat` file contains actual data.

# COMMAND ----------

!cat /databricks/driver/cap1f_01.cfg

# COMMAND ----------

!cat /databricks/driver/cap1f_01.dat

# COMMAND ----------

# MAGIC %md And we can display the contents of a sample file as a graph:

# COMMAND ----------

# DBTITLE 1,Plot a Sample File
from comtrade import Comtrade
import matplotlib.pyplot as plt

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

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | comtrade | A module designed to read Common Format for Transient Data Exchange (COMTRADE) file format |  MIT | https://pypi.org/project/comtrade/                       |
# MAGIC | comtradehandlers | File handlers for the COMTRADE format| MIT | https://github.com/relihanl/comtradehandlers.git#egg=comtradehandlers |

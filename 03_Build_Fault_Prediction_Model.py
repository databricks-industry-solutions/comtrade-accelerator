# Databricks notebook source
# MAGIC %md The purpose of this notebook is to train a fault prediction model as part of the COMTRADE solution accelerator.  This notebook was developed on a **Databricks ML 12.2 LTS** cluster. This notebook is also available at https://github.com/databricks-industry-solutions/comtrade-accelerator.

# COMMAND ----------

# DBTITLE 0,ion
# MAGIC %md ##Introduction
# MAGIC 
# MAGIC With our COMTRADE data now recorded in a more accessible format, we can turn our attention to applications for this data.  While there are many, many ways to make use of COMTRADE data, one of the more common uses is in fault prediction.
# MAGIC 
# MAGIC In this notebook, we'll tackle fault prediction using a binary classification approach where we identify a set of readings as associated with a fault (or not). We will use a convolutional neural network (CNN) pattern that's become popularized as of late for its ability to understand the more complex patterns of relationships within a waveform signal.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

import tensorflow as tf
import mlflow
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_Intro_and_Config"

# COMMAND ----------

# MAGIC %md ##Step 1: Get Features & Labels
# MAGIC 
# MAGIC Before retrieving the values associated with our readings, it's important for us to establish the number of readings in each file.  We will be training a model on the patterns found in each using a signal processing approach that expects a consistent set of input values. 
# MAGIC 
# MAGIC From an examination of all the files in this dataset, we can see each contains 726 readings.  We can set our signal *frame size* to 726 or some evenly divisible unit of this, *i.e.* 363, 242, 121, 66, 33, 22, 11, 6, 3, or 2.
# MAGIC .

# COMMAND ----------

# DBTITLE 1,Examine Number of Readings in Files
readings = (
  spark
    .table('readings')
    .groupBy('path_dat')
      .agg(fn.count('*').alias('readings'))
    .groupBy()
      .agg(
        fn.min('readings').alias('min_readings'),
        fn.max('readings').alias('max_readings')
        )
  )

display(readings)

# COMMAND ----------

# DBTITLE 1,Capture Readings per File
readings_per_file = 726

# COMMAND ----------

# MAGIC %md We can now retrieve our features and labels.  In this model, our features will be all 726 readings associated with each file.  Our label will be set to 1 if the file represents a fault (as given in the file's name) and 0 if it does not:  

# COMMAND ----------

# DBTITLE 1,Assemble Features and Labels
inputs_pd = (
  spark
    .table('metadata').alias('cfg')
    .withColumn('is_fault', fn.expr("cast(contains(path_cfg,'external') as int)")) # 1 if name contains word 'external', otherwise 0
    .join(
        spark.table('readings').alias('dat'),
        on=['path_cfg','path_dat']
      )
    .orderBy('cfg.path_cfg','dat.microseconds')
    .selectExpr(
      'is_fault',
      'IA',
      'IB',
      'IC'
      )
  ).toPandas()

display(inputs_pd)

# COMMAND ----------

# MAGIC %md Our model will be trained using Tensorflow signal frames, a popular structure in ML-based signal processing that organizes all the readings to be analyzed as a unit within a frame.  To keep things simple, we'll lump all 726 readings for each individual file into a frame that will be considered in its entirety.  Per the discussion above, we could divide this into overlapping or non-overlapping units so long as the readings for one file are not loaded into a frame with those of another:

# COMMAND ----------

# DBTITLE 1,Define Tensors to Hold Features & Labels
# assemble signal frame features
features_tensor = (
  tf.signal.frame(
    inputs_pd.loc[:, ['IA', 'IB', 'IC']].to_numpy(), # input signal data
    readings_per_file, # read 726 inputs together
    readings_per_file, # next reading at the 726 mark to avoid overlap
    axis=0
    )
  )

# assemble signal frame labels
labels_tensor = (
  tf.signal.frame(
    inputs_pd['is_fault'].iloc[::readings_per_file].to_numpy(), 
    1, 
    1
    )
  )

# Print the shapes
print('Features: ', features_tensor.shape)
print('Labels:   ', labels_tensor.shape)

# COMMAND ----------

# MAGIC %md Because of how we assigned labels to our files, once we encounter a frame labeled as a fault, we are likely to have several that follow it that are also faults (simply because these files are in a directory containing only faults).  By randomly shuffling the frames, we can help ensure the model doesn't inadvertently learn that fault files tend to go together or otherwise over optimize to the prediction of any one class at a given time:

# COMMAND ----------

# DBTITLE 1,Shuffle Features & Labels
# get indices for each frame
indices = tf.range(
  start=0, 
  limit=tf.shape(features_tensor)[0], 
  dtype=tf.int32
  )

# shuffle the indices
shuffled_indices = tf.random.shuffle(indices)

# reorganize the data around the shuffled indices
shuffled_signals = tf.gather(features_tensor, shuffled_indices)
shuffled_labels = tf.gather(labels_tensor, shuffled_indices)

# COMMAND ----------

# MAGIC %md With our data randomly shuffled, we can now split it into training, validation and test sets as follows:

# COMMAND ----------

# DBTITLE 1,Perform Train, Validate & Test Splits
# identify indices to split data
train_stop = int(shuffled_signals.shape[0] * 0.7) # break at 70%
val_stop = int(shuffled_signals.shape[0] * (0.7 + 0.15)) # next 15% goes to val
# remainder goes to test

# train
train_signals = shuffled_signals[:train_stop]
train_labels = shuffled_labels[:train_stop]

# validation
val_signals = shuffled_signals[train_stop:val_stop]
val_labels = shuffled_labels[train_stop:val_stop]

# test
test_signals = shuffled_signals[val_stop:]
test_labels = shuffled_labels[val_stop:]

# print row counts for validation
print('Training:   ', train_signals.shape[0])
print('Validation: ', val_signals.shape[0])
print('Testing:    ', test_signals.shape[0])

# COMMAND ----------

# MAGIC %md ##Step 2: Define the Model
# MAGIC 
# MAGIC  Now we can define our model.  We'll use a simple convolutional neural network (CNN) architecture that's frequently employed in signals prediction.  You can read more about the application of CNNs in this space in [this](https://ieeexplore.ieee.org/document/8771146) and [related documents](https://www.hindawi.com/journals/itees/2022/8431450/):

# COMMAND ----------

# DBTITLE 1,Define Model for Fault Prediction
def create_convolutional_classification_model() -> tf.keras.Model:
    
    #tf.random.set_seed(13)

    inp = tf.keras.Input(shape=[readings_per_file,3])
    pipe = tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same") (inp)
    pipe = tf.keras.layers.MaxPooling1D(pool_size=4) (pipe)
    pipe = tf.keras.layers.Conv1D(32,3, activation="relu", padding="same") (pipe)
    pipe = tf.keras.layers.MaxPooling1D(pool_size=4) (pipe)
    pipe = tf.keras.layers.Flatten() (pipe)
    pipe = tf.keras.layers.Dropout(0.5) (pipe)
    pipe = tf.keras.layers.Dense(1, activation="sigmoid") (pipe)

    mod = tf.keras.Model(inp,pipe)
    mod.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["Precision","Recall"])
    return mod

# COMMAND ----------

# MAGIC %md ##Step 3: Tune the Model
# MAGIC 
# MAGIC Once the model architecture is defined, we can tune the settings that control how the model learns from our data. The key values are *batch_size* and *epochs*.  *batch_size* determines how many values are consumed at a time as the model is processed and *epochs* controls how many total passes over the data the model will take as part of the overall learning process.
# MAGIC 
# MAGIC Because we don't know exactly what the right values for these two hyperparameters should be, we can define a search space over which acceptable values are likely to be found:

# COMMAND ----------

# DBTITLE 1,Define Hyperparameter Search Space
search_space = {
    'batch_size' : hp.quniform('batch_size', 10.0, 100.0, 1.0)                       
    ,'epochs' : hp.quniform('epochs', 5.0, 20.0, 1.0)   
    }

# COMMAND ----------

# MAGIC %md We can then define a function that when passed a set of values from this search space will train a model.  This function will be used in a later step to evaluate a large number of different hyperparameter value combinations.  The function returns a loss (error) value once it is done training a given model.  Our goal will be to find a set of hyperparameter values that minimizes loss:

# COMMAND ----------

# DBTITLE 1,Define Function to Evaluate Model against Given Hyperparameter Values
def evaluate_model(hyperopt_params):

    _train_signals = train_signals_broadcast.value
    _train_labels = train_labels_broadcast.value
    _val_signals = val_signals_broadcast.value
    _val_labels = val_labels_broadcast.value

    params = hyperopt_params
    if 'batch_size' in params: params['batch_size']=int(params['batch_size'])   # hyperopt supplies values as float but must be int
    if 'epochs' in params: params['epochs']=int(params['epochs']) # hyperopt supplies values as float but must be int
    # all other hyperparameters are taken as given by hyperopt

    # instantiate model
    model = create_convolutional_classification_model()

    # fit the model
    model_history = (
      model.fit(
        x= _train_signals, 
        y= _train_labels, 
        validation_data=( _val_signals, _val_labels),
        **params
        )
      )

    # get evaulation metrics
    loss = model_history.history['val_loss'][-1]
    precision = model_history.history['val_precision'][-1]
    recall = model_history.history['val_recall'][-1]
    
    # log metrics
    mlflow.log_metrics({
      'loss':loss,
      'precision':precision,
      'recall':recall
      })

    return {'loss':loss, 'status':STATUS_OK}

# COMMAND ----------

# MAGIC %md The function above depends on access to our training and validation datasets.  Because we will be calling this function on the different nodes that make up our Databricks cluster, we can improve the performance of each function call be creating an in-memory copy of each dataset on the server nodes that make up our cluster.  We do this with a simple broadcast operation:

# COMMAND ----------

# DBTITLE 1,Broadcast Variables
train_signals_broadcast = sc.broadcast(train_signals)
train_labels_broadcast = sc.broadcast(train_labels)
val_signals_broadcast = sc.broadcast(val_signals)
val_labels_broadcast = sc.broadcast(val_labels)

# COMMAND ----------

# MAGIC %md And now we can tune our model.  Note that we are using a library called [Hyperopt](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/index.html) to perform this work.  Hyperopt as configured here will coordinate the distribution of some number of parallel trail runs, selecting values from the available search space and passing a unique combination of values to each of these parallel trails.  
# MAGIC 
# MAGIC After each trail is done with its work, Hyperopt will evaluate the loss value returned by each and use that information to constrain the search space.  Using this constrained search space, it triggers the next wave of trails to continue constraining the space until the max number of evaluations is reached.  At that point, the hyperparameter combinations that have returned the best results will be determined to be our best hyperparameter values.
# MAGIC 
# MAGIC Setting *max_evals* to a higher value will cause this process to take longer but is likely to provide you a more fine-tuned search that may result in better hyperparameter values.  Using Hyperopt (in combination with smart sizing of your Databricks cluster), you can perform an intelligent search of a hyperparameter search space within your given time-constraints:
# MAGIC 
# MAGIC **NOTE** We'll return to the mlflow stuff in Step 4.

# COMMAND ----------

# DBTITLE 1,Identify the Optimal Hyperparameter Values
# perform evaluation
with mlflow.start_run(run_name='tuning'):
  
  argmin = fmin(
    fn=evaluate_model,
    space=search_space,
    algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
    max_evals=20,
    trials=SparkTrials(parallelism=4),  # 20 evals with 4 evals at a time = 5 learning cycles
    verbose=True
    )
  
# separate hyperopt output from our results
print('\n')

# capture optimized hyperparameters
hyperopt_params = space_eval(search_space, argmin)
hyperopt_params

# COMMAND ----------

# MAGIC %md ##Step 4: Train the Model
# MAGIC 
# MAGIC With optimal hyperparameter values identified, we can now proceed with the training of our final model.  For this round of training, we can combine our training and validation datasets (used in the hyperparameter tuning exercise) to serve as our new training set: 

# COMMAND ----------

# DBTITLE 1,Combine Training & Validation Sets for Final Training
train_val_signals = tf.concat([train_signals, val_signals],axis=0)
train_val_labels = tf.concat([train_labels, val_labels],axis=0)

# COMMAND ----------

# MAGIC %md And now we can train our final model.  Please note that this is largely a copy-and-paste of code used in the *evaluate_model* function, but we are not using the broadcast feature sets as this will run once on the driver node in our cluster where the original datasets are already resident in memory:

# COMMAND ----------

# DBTITLE 1,Train Final Model
with mlflow.start_run(run_name='final') as run:

  # instantiate model
  model = create_convolutional_classification_model()

  # make sure hyperparams are correct data types
  params = hyperopt_params
  if 'batch_size' in params: params['batch_size']=int(params['batch_size']) # hyperopt supplies values as float but must be int
  if 'epochs' in params: params['epochs']=int(params['epochs']) # hyperopt supplies values as float but must be int

  # fit the model
  model_history = (
    model.fit(
      x= train_val_signals, 
      y= train_val_labels, 
      validation_data=( test_signals, test_labels),
      **params
      )
    )

  # get evaulation metrics
  loss = model_history.history['val_loss'][-1]
  precision = model_history.history['val_precision'][-1]
  recall = model_history.history['val_recall'][-1]

  # log metrics
  mlflow.log_metrics({
  'loss':loss,
  'precision':precision,
  'recall':recall
  })

  # log model
  mlflow.tensorflow.log_model(
    model, 
    artifact_path='model',
    registered_model_name=config['model_name']
    )

print('Precision: ',precision)
print('Recall:    ',recall)
print('Loss:      ',loss)

# COMMAND ----------

# MAGIC %md Notice as part of this last exercise, we trained our model as part of an mlflow run.  (We did the same thing when we performed hyperparameter tuning but didn't explain it at that time in order to keep the focus on Hyperopt.)  MLflow is an open source technology that's integrated into the Databricks environment.  It allows us to perform model tracking, registration and deployment, all of which are actions that help ensure we can better manage our model development work and more easily move models into production deployments.  You can read more about mlflow [here](https://docs.databricks.com/mlflow/index.html).
# MAGIC 
# MAGIC Examining the code above, you can see we trained our model as part of an mlflow run.  We can now search our runs to locate the latest instance of this model and elevate it to production status.  This step is more typically performed using the MLflow user interface follow a series of evaluations performed by our machine learning and develop operations personnel and automation packages.  We are simply moving this into production now to paint a more complete picture of how this model may be identified as *ready for production use*:

# COMMAND ----------

# DBTITLE 1,Move Model to Production Status
client = mlflow.tracking.MlflowClient()
model_version = client.search_model_versions(f"name='fault_detection'")[0].version

# move model version to production
client.transition_model_version_stage(
  name=config['model_name'],
  version=model_version,
  stage='production',
  archive_existing_versions = True
  )      

# COMMAND ----------

# MAGIC %md ##Step 5: Deploy the Model
# MAGIC 
# MAGIC With our model in MLflow, it can easily be deployed in a variety of scenarios.  The one that most immediately comes to mind is one within which COMTRADE data are delivered from electrical providers as part of their fault management processes.  These files may be processed upon receipt in real-time using Databricks [Auto Loader](https://docs.databricks.com/ingestion/auto-loader/index.html) and [Delta Live Table](https://docs.databricks.com/delta-live-tables/index.html) logic that persists the data to Delta Lake tables (as performed in the previous notebook) and [presented to the latest production instance of our fault detection model](https://docs.databricks.com/delta-live-tables/transform.html#use-mlflow-models-in-a-delta-live-tables-pipeline) to determine if a fault has occurred. From there, Databricks may send a message to any number of [downstream systems](https://docs.databricks.com/external-data/index.html) in order to notify them of the occurrence.
# MAGIC </p>
# MAGIC 
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/images/comtrade_architecture2.png" width=60%>
# MAGIC </p>

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | comtrade | A module designed to read Common Format for Transient Data Exchange (COMTRADE) file format |  MIT | https://pypi.org/project/comtrade/                       |
# MAGIC | comtradehandlers | File handlers for the COMTRADE format| MIT | https://github.com/relihanl/comtradehandlers.git#egg=comtradehandlers |

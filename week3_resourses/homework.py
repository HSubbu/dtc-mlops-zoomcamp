import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

import datetime
import requests
import pickle

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta
from prefect.orion.schemas.schedules import CronSchedule

@task
def get_paths(date_time_str):
    output_directory = "./data"
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
    data_month = date_time_obj.month
    data_year = date_time_obj.year
    train_month = data_month-2
    val_month = data_month-1
    if train_month < 10:
        train_path = f'./data/fhv_tripdata_{data_year}-0{train_month}.parquet'
    else:
        train_path = f'./data/fhv_tripdata_{data_year}-{train_month}.parquet'
    if val_month <10:
        val_path = f'./data/fhv_tripdata_{data_year}-0{val_month}.parquet'
    else:
        val_path = f'./data/fhv_tripdata_{data_year}-{val_month}.parquet'
    
    if train_month <10:
        train_url=f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{data_year}-0{train_month}.parquet"
    else:
        train_url=f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{data_year}-{train_month}.parquet"
    
    if val_month<10:
        val_url=f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{data_year}-0{val_month}.parquet"
    else:
        val_url=f"https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{data_year}-{val_month}.parquet"
    
    response_train = requests.get(train_url)
    open(train_path, "wb").write(response_train.content)

    response_val = requests.get(val_url)
    open(val_path, "wb").write(response_val.content)



    return train_path,val_path

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    with open(f'model-{date}.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)
    with open(f'dv-{date}.bin', 'wb') as f_out:
        pickle.dump(dv, f_out)

DeploymentSpec(
  flow=main,
  name="model_training",
  schedule=CronSchedule(cron="0 9 15 * *",timezone="America/New_York"),
  flow_runner=SubprocessFlowRunner(),
  tags=["ml"]
)

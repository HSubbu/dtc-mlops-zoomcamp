#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import argparse

import os
year = os.environ['year']
month = os.environ['month']
print(year,month)

#parser = argparse.ArgumentParser()
#parser.add_argument('--year' , help="year_input")
#parser.add_argument('--month' , help="month_input")
#args = parser.parse_args()
#year=args.year
#month=args.month
#print(f'Passed arguments for year and month are {args.year},{args.month}')


with open('model.bin', 'rb') as f_in:
    dv,lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

url = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year}-0{month}.parquet'
print(url)

df = read_data(url)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

print(f'mean predicted duration {y_pred.mean()}')


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()

df_result['ride_id'] = df['ride_id']

df_result['predictions'] = pd.Series(y_pred)

output_file= "./outputs"

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)






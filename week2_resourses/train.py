import argparse
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow

import warnings
warnings.filterwarnings('ignore')

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "nyc-taxi-homework"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

mlflow.sklearn.autolog()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_valid)

    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()
    with mlflow.start_run():
        mlflow.set_tag("developer", "Subramanian")
        run(args.data_path)
        #mlflow.log_metric("rmse", run_rmse)

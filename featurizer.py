"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import yaml 

# import boto3
import numpy as np
import pandas as pd

from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    # logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-schema-path", type=str, default="schema.yml")
    parser.add_argument('--target-col', type=str) 
    parser.add_argument('--train-path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--train-file', type=str, default='boston_train.csv')
    # parser.add_argument('--test-path', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    # parser.add_argument('--test-file', type=str, default='boston_test.csv')
    # parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    # parser.add_argument(
    #     "--aws-profile",
    #     dest="aws_profile",
    #     type=str,
    #     help="The AWS profile that will be used.",
    #     default="default",
    # )
    # parser.add_argument(
    #     "--raw-bucket",
    #     dest="raw_bucket",
    #     type=str,
    #     help="The s3 bucket containing the raw data.",
    #     required=True
    # )
    # parser.add_argument(
    #     "--tmp-data-dir",
    #     dest="tmp_data_dir",
    #     type=str,
    #     help="The temporary local directory where raw, split, transformed data will be stored.",
    #     default="tmp/data/"
    # )
    
    args = parser.parse_args()

    # base_dir = "opt/ml/processing"
    # pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    # input_data = args.input_data
    # bucket = input_data.split("/")[2]
    # key = "/".join(input_data.split("/")[3:])

    # logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    # fn = f"{base_dir}/data/abalone-dataset.csv"
    # my_session = boto3.session.Session(profile_name=args.aws_profile)
    # s3_resource = my_session.resource('s3')
    # s3_resource.Bucket(bucket).download_file(key, fn)

    # logger.debug("Reading downloaded data.")
    with open(args.raw_schema_path) as file:
        raw_schema = yaml.load(file, Loader=yaml.FullLoader)

    # X_train = pd.read_csv(
    #     args.tmp_data_dir + "split/X_train.csv",
    #     dtype=dtype
    # )
    # X_val = pd.read_csv(
    #     args.tmp_data_dir + "split/X_val.csv",
    #     dtype=dtype
    # )
    # X_test = pd.read_csv(
    #     args.tmp_data_dir + "split/X_test.csv",
    #     dtype=dtype
    # )

    print('reading data')
    train_df = pd.read_csv(os.path.join(args.train_path, args.train_file), dtype=raw_schema)
    # test_df = pd.read_csv(os.path.join(args.test_path, args.test_file), dtype=raw_schema)

    print('building training and testing datasets')
    X_train = train_df.copy()
    y_train = train_df.pop(args.target_col)
    # X_test = test_df.copy()
    # y_test = test_df.pop(args.target_col)

    # logger.debug("Defining transformers.")
    feature_cols = list(X_train.columns)
    numeric_features = feature_cols[:]
    numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = ["sex"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    featurizer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    featurizer.fit(X_train)


    # logger.info("Applying transforms.")
    # X_train = preprocess.fit_transform(X_train)
    # X_val = preprocess.transform(X_val)
    # X_test = preprocess.transform(X_test)

    # df_train = np.concatenate((y_train.to_numpy().reshape(-1, 1), X_train), axis=1)
    # df_validation = np.concatenate((y_validation.to_numpy().reshape(-1, 1), X_validation), axis=1)
    # df_test = np.concatenate((y_test.to_numpy().reshape(-1, 1), X_test), axis=1)

    # logger.info("Writing out datasets to %s and s3.", args.tmp_data_dir)
    # pathlib.Path(args.tmp_data_dir + "transformed/").mkdir(parents=True, exist_ok=True)
    # for channel, X_ in [('train', X_train), ('val', X_val), ('test', X_test)]:
    #     np.savetxt(
    #         args.tmp_data_dir + "transformed/X_" + channel + ".csv",
    #         X_,
    #         delimiter=",",
    #     )
    #     s3_resource.Bucket(args.raw_bucket).upload_file(
    #         args.tmp_data_dir + "transformed/X_" + channel + ".csv",
    #         "data/transformed/X_" + channel + ".csv",
    #     )

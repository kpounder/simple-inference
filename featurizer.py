# TODO
# Try to read in schema from a file as opposed to having it hard coded here 
# Do I need to save the model here via joblib -- will the SKLearn fit method already do that? 


"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import yaml 
import joblib
from io import StringIO

import boto3
import numpy as np
import pandas as pd

from sagemaker_containers.beta.framework import (
    content_types,
    encoders,
    env,
    modules,
    transformer,
    worker,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

logger.debug("Reading in schema.")
feature_cols_dtype = {
    'sex': str,
    'length': float,
    'diameter': float,
    'height': float,
    'whole_weight': float,
    'shucked_weight': float,
    'viscera_weight': float,
    'shell_weight': float,
}
feature_cols = list(feature_cols_dtype.keys())
target_col_dtype = {
    'rings': float
}
target_col = list(target_col_dtype.keys())[0]

# with open("schema.yml") as file:
#     raw_schema = yaml.load(file, Loader=yaml.FullLoader)
# target_col = raw_schema['target_col']
# target_col_dtype = raw_schema['dtype'][target_col]
# feature_cols_dtype = raw_schema['dtype']
# del(feature_cols_dtype[target_col])
# feature_cols = list(feature_cols_dtype.keys())


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-schema-path", type=str, default="schema.yml")
    parser.add_argument('--target-col', type=str, default="rings") 
    parser.add_argument('--X-train-path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--X-train-file', type=str, default='X_train.csv')
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
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

    # print('building training and testing datasets')
    logger.debug("Reading in X_train.")
    X_train = pd.read_csv(
        os.path.join(args.X_train_path, args.X_train_file), names=feature_cols, dtype=feature_cols_dtype
    )

    # logger.debug("Defining transformers.")
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
    # featurizer = Pipeline(steps=[('ct', ct)])

    logger.debug("Fitting featurizer.")
    featurizer.fit(X_train)
    
    logger.debug("Saving model")
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(featurizer, path)


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

    
def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None)

        if len(df.columns) == len(feature_cols) + 1:
            # This is a labelled example, includes the ring label
            df.columns = feature_cols + [target_col]
        elif len(df.columns) == len(feature_cols):
            # This is an unlabelled example.
            df.columns = feature_cols

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == "text/csv":
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    if target_col in input_data:
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data[target_col], axis=1)
    else:
        # Return only the set of features
        return features


def model_fn(model_dir):
    """Deserialize fitted model"""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
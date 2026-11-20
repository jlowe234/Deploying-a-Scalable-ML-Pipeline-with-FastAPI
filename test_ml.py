import pytest
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model
)

def test_model_is_logistic_regression():
    """
    Verify that the model is a logistic regression model
    """
    path = os.path.join('./', 'model', 'model.pkl')
    model = load_model(path)

    assert type(model) == LogisticRegression


def test_metrics_are_correct_for_model():
    """
    Verify that the metrics of the model are as expected
    """

    # Given
    project_path = "./"
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.3, random_state=234)
    path = os.path.join('./', 'model', 'model.pkl')
    model = load_model(path)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    _, _, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary"
        )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # When
    preds = inference(model, X_test)
    p, r, fb = compute_model_metrics(y_test, preds)

    # Then
    assert pytest.approx(p, abs=0.0001) == 0.7452
    assert pytest.approx(r, abs=0.0001) == 0.6056
    assert pytest.approx(fb, abs=0.0001) == 0.6682

def test_training_data_is_sized_correctly():
    """
    Validate the size of the training data set
    """
    # Given
    project_path = "./"
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)

    # When
    train, _ = train_test_split(data, test_size=0.3, random_state=234)

    # Then
    assert train.shape == (22792, 15)

def test_testing_data_is_sized_correctly():
    """
    Validate the size of the testing data set
    """
    # Given
    project_path = "./"
    data_path = os.path.join(project_path, "data", "census.csv")
    data = pd.read_csv(data_path)

    # When
    _ , test = train_test_split(data, test_size=0.3, random_state=234)

    # Then
    assert test.shape == (9769, 15)
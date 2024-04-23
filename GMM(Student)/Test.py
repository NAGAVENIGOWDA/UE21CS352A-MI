import sys
import importlib
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ID", required=True)

args = parser.parse_args()
subname = args.ID

try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print("Rename your written program as GMM.py and run python Test.py --ID GMM")
    sys.exit()

GMMModel = mymodule.GMMModel


def generate_data():
    age = torch.tensor(
        [35, 45, 30, 28, 40, 55, 32, 48, 50, 42, 29, 37, 52, 31]
    ).unsqueeze(1)
    income = torch.tensor(
        [
            60000,
            75000,
            50000,
            55000,
            70000,
            80000,
            45000,
            90000,
            95000,
            60000,
            48000,
            62000,
            80000,
            55000,
        ]
    ).unsqueeze(1)
    purchase_frequency = torch.tensor(
        [2, 4, 1, 3, 5, 6, 2, 4, 3, 2, 1, 5, 4, 3]
    ).unsqueeze(1)
    dataset = torch.cat((age, income, purchase_frequency), dim=1)

    return dataset


def test_case():
    # Generate customer segmentation data
    data = generate_data()

    try:
        # Test GMM fitting
        gmm = GMMModel(n_components=3)
        gmm.fit(data)

        # Test GMM prediction
        labels = gmm.predict(data)
        if labels.shape == (data.shape[0],):
            print("Test Case 1 for GMM fitting and prediction PASSED")
        else:
            print("Test Case 1 for GMM fitting and prediction FAILED")

        # Test getting cluster means
        cluster_means = gmm.get_cluster_means()
        if cluster_means.shape == (3, data.shape[1]):
            print("Test Case 2 for getting cluster means PASSED")
        else:
            print("Test Case 2 for getting cluster means FAILED")

        # Test getting cluster covariances
        cluster_covariances = gmm.get_cluster_covariances()
        if cluster_covariances.shape == (3, data.shape[1], data.shape[1]):
            print("Test Case 3 for getting cluster covariances PASSED")
        else:
            print("Test Case 3 for getting cluster covariances FAILED")

        # Test GMM prediction for a new sample
        new_sample = torch.tensor([[35, 60000, 2]], dtype=torch.float32)
        predicted_label = gmm.predict(new_sample)
        if predicted_label.shape == (1,) and 0 <= predicted_label[0] < 3:
            print("Test Case 4 for GMM prediction PASSED")
        else:
            print("Test Case 4 for GMM prediction FAILED")
    except Exception as e:
        print("Test Case 1, 2, 3, or 4 FAILED [ERROR]:", str(e))


if __name__ == "__main__":
    test_case()



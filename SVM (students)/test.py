import sys
import importlib
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--ID', required=True)

args = parser.parse_args()
subname = args.ID


try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print(e)
    print("Rename your written program as per specified format and run 'python3 test.py --ID file_name' ")
    sys.exit()

dataset_helper = mymodule.DatasetHelper()

train_dataset_1_path = './train1.csv'
train_dataset_2_path = './train2.csv'
train_dataset_3_path = './train3.csv'
train_dataset_paths = [train_dataset_1_path, train_dataset_2_path, train_dataset_3_path]

try:
    datasets = [dataset_helper.read(dataset_path) for dataset_path in train_dataset_paths]
    svm = mymodule.SVM(datasets)
  
    svm.build_models()

    # train dataset paths are specified below so that you can test your solutions.
    # During evaluation, these paths will be changed to those for the test datasets.
    test_dataset_1_path = './train1.csv'
    test_dataset_2_path = './train2.csv'
    test_dataset_3_path = './test3_visible.csv'
    test_dataset_paths = [test_dataset_1_path, test_dataset_2_path, test_dataset_3_path]

    for i, model in enumerate(svm.models):
        try:
            dataset = dataset_helper.read(test_dataset_paths[i])
            if i != 0:
                print(f'Accuracy: {model.score(dataset.X, dataset.y) * 100:.2f}%')
            else:
                y_pred = model.predict(dataset.X)
                print(f'MSE: {mean_squared_error(dataset.y, y_pred):.3f}')
        except Exception as e:
            print(f'Model Failed {e}')
except Exception as e:
    print(f'Failed {e}')

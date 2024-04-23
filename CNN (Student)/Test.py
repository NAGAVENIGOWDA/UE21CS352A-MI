import sys
import importlib
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--ID", required=True)

args = parser.parse_args()
subname = args.ID


try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print(
        "Rename your written program as  CAMPUS_SECTION_SRN_Lab3.py and run python Test.py --ID  CAMPUS_SECTION_SRN_Lab3 "
    )
    sys.exit()

prepareData = mymodule.prepareData
CNN = mymodule.CNN
train = mymodule.train
evaluate = mymodule.evaluate

def test_case():
   
    train_loader,test_loader = prepareData([1,8,9],{
        1:0,
        8:1,
        9:2
    })
    model = CNN()
    # Train the model
    train_accuracy = train(model,train_loader)
    
    # Evaluate the model on the test set
    test_accuracy = evaluate(model,test_loader)
    
    try: 
        if train_accuracy>=60:
            print("Test Case 1 for the function train PASSED")
        else:
            print("Test Case 1 for the function train FAILED")
    except Exception as e:
        print(e)
        print("Test Case 1 for the function train FAILED [ERROR]")

    try: 
        if test_accuracy>=70:
            print("Test Case 2 for the function evaluate PASSED")
        else:
            print("Test Case 2 for the function evaluate FAILED")
    except Exception as e:
        print(e)
        print("Test Case 2 for the function evaluate FAILED [ERROR]")

    try:
        if(len(list(model.children())) <=9):
            print("Test Case 3 for the function __init__ PASSED")
        else:
            print("Test Case 3 for the function __init__ FAILED")
    except  Exception as e:
        print(e)
        print("Test Case 3 for the function __init__ FAILED [ERROR]")

if __name__ == "__main__":
    test_case()

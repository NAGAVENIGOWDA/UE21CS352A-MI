import sys
import importlib
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ID", required=True)

args = parser.parse_args()
subname = args.ID


try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print(
        "Rename your written program as  CAMPUS_SECTION_SRN_Lab1.py and run python Test.py --ID  CAMPUS_SECTION_SRN_Lab1 "
    )
    sys.exit()


get_selected_attribute = mymodule.get_selected_attribute
get_information_gain = mymodule.get_information_gain
get_avg_info_of_attribute = mymodule.get_avg_info_of_attribute
get_entropy_of_dataset = mymodule.get_entropy_of_dataset


def test_case():
    outlook = torch.tensor([1,1,1,1,2,2,2,2,2,3,3,3,3,3]).unsqueeze(1)
    temp = torch.tensor([1,2,3,1,3,2,2,3,3,1,1,3,2,3]).unsqueeze(1)
    humidity = torch.tensor([1,2,1,2,1,2,2,2,1,1,1,1,2,2]).unsqueeze(1)
    windy = torch.tensor([0,1,1,0,0,0,1,0,1,0,1,0,0,1]).unsqueeze(1)
    play = torch.tensor([1,1,1,1,1,1,0,1,0,0,0,0,1,1]).unsqueeze(1)
    dataset = torch.cat((outlook,temp,humidity,windy,play),dim=1)

    try:
        if get_entropy_of_dataset(dataset) >= 0.938 and get_entropy_of_dataset(dataset) <= 0.942:
            print("Test Case 1 for the function get_entropy_of_dataset PASSED")
        else:
            print("Test Case 1 for the function get_entropy_of_dataset FAILED")
    except:
        print("Test Case 1 for the function get_entropy_of_dataset FAILED [ERROR]")

    try:
        if (
            get_avg_info_of_attribute(dataset, 0) >= 0.691
            and get_avg_info_of_attribute(dataset, 0) <= 0.695
        ):
            print("Test Case 2 for the function get_avg_info_of_attribute PASSED")
        else:
            print("Test Case 2 for the function get_avg_info_of_attribute FAILED")

    except:
        print("Test Case 2 for the function get_avg_info_of_attribute FAILED [ERROR]")

    try:
        
        if (
            get_avg_info_of_attribute(dataset, 1) >= 0.908
            and get_avg_info_of_attribute(dataset, 1) <= 0.914
        ):
            print("Test Case 3 for the function get_avg_info_of_attribute PASSED")
        else:
            print("Test Case 3 for the function get_avg_info_of_attribute FAILED")

    except:
        print("Test Case 3 for the function get_avg_info_of_attribute FAILED [ERROR]")

    try:
        columns = [0, 1, 2, 3, 4]
        ans = get_selected_attribute(dataset)
        dictionary = ans[0]
        flag = (
            (dictionary[0] >= 0.244 and dictionary[0] <= 0.248)
            and (dictionary[1] >= 0.0292 and dictionary[1] <= 0.0296)
            and (dictionary[2] >= 0.150 and dictionary[2] <= 0.154)
            and (dictionary[3] >= 0.046 and dictionary[3] <= 0.05)
            and (ans[1] == 0)
        )
        if flag:
            print("Test Case 4 for the function get_selected_attribute PASSED")
        else:
            print("Test Case 4 for the function get_selected_attribute FAILED")
    except:
        print("Test Case 4 for the function get_selected_attribute FAILED [ERROR]")


if __name__ == "__main__":
    test_case()



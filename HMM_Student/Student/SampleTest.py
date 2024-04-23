import sys
import importlib
import argparse
import torch 


parser = argparse.ArgumentParser()
parser.add_argument('--SRN', required=True)

args = parser.parse_args()
subname = args.SRN


try:
    mymodule = importlib.import_module(subname)
except Exception as e:
    print(e)
    print("Rename your written program as YOUR_SRN.py and run python3.7 SampleTest.py --SRN YOUR_SRN ")
    sys.exit()

HMM = mymodule.HMM


def test_1():
    '''
    Bob's observed mood (Happy or Grumpy) 
    can be modelled with the weather (Sunny, Rainy)
    '''
    A = torch.tensor([
        [0.8, 0.2],
        [0.4, 0.6]
    ])

    HS = ['Sunny', 'Rainy']
    O = ["Happy", 'Grumpy']
    priors = [2/3, 1/3]

    B = torch.tensor([
        [0.8, 0.2],
        [0.4, 0.6]
    ])

    ES = ["Happy", "Grumpy", "Happy"]
    model = HMM(A, HS, O, priors, B)
    seq = model.viterbi_algorithm(ES)
    assert (seq == ['Sunny', 'Sunny', 'Sunny'])


def test_2():
    ''' Bob's observation mod (Happy or Sad) can be modelled with his hobbies (Movie,Book, Party) '''  
    A = torch.tensor([
        [0.9, 0.1],
        [0.01, 0.99]
    ])

    HS = ['Happy', 'Sad']
    O = ['Movie', 'Book','Party']
    priors = [0.2, 0.8]
    B = torch.tensor([
        [0.3, 0.3,0.4],
        [0.5, 0.3,0.2]
    ])
    ES = ['Movie', 'Book', 'Party']
    model = HMM(A, HS, O, priors, B)
    seq = model.viterbi_algorithm(ES)
    assert(seq == ['Sad', 'Sad', 'Sad'])


def test_3():
    A = torch.tensor([
        [0.2, 0.8],
        [0.7, 0.3]
    ])

    HS = ['A', 'B']
    O = ['x', 'y']
    priors = [0.7, 0.3]
    B = torch.tensor([
        [0.4, 0.6],
        [0.3, 0.7]
    ])
    ES = ['x', 'y', 'y']
    model = HMM(A, HS, O, priors, B)
    seq = model.viterbi_algorithm(ES)
    assert(seq == ['A', 'B', 'A'])


if __name__ == "__main__":
    try:
        test_1()
        print("Test case 1 for Viterbi Algorithm passed!")
    except Exception as e:
        print(f"Test case 1 for Viterbi Algorithm failed!\n{e}")

    try:
        test_2()
        print("Test case 2 for Viterbi Algorithm passed!")
    except Exception as e:
        print(f"Test case 2 for Viterbi Algorithm failed!\n{e}")

    try:
        test_3()
        print("Test case 3 for Viterbi Algorithm passed!")
    except Exception as e:
        print(f"Test case 3 for Viterbi Algorithm failed!\n{e}")

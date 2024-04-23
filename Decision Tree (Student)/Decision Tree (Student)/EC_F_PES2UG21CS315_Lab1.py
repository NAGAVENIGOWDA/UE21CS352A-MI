import math

import torch


def calculate_entropy(probabilities):
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

def get_entropy_of_dataset(data_tensor):
    class_column = data_tensor[:, -1]
    unique_classes, class_counts = torch.unique(class_column, return_counts=True)
    total_samples = data_tensor.shape[0]
    class_probs = class_counts.float() / total_samples
    entropy = calculate_entropy(class_probs)
    return entropy

def get_avg_info_of_attribute(data_tensor, attribute_index):
    unique_vals, value_counts = torch.unique(data_tensor[:, attribute_index], return_counts=True)
    total_samples = data_tensor.shape[0]
    
    avg_info = 0
    for value, count in zip(unique_vals, value_counts):
        subset = data_tensor[data_tensor[:, attribute_index] == value]
        subset_ent = get_entropy_of_dataset(subset)
        value_prob = count / total_samples
        avg_info += value_prob * subset_ent
    
    return avg_info

def get_information_gain(data_tensor, attribute_index):
    dataset_ent = get_entropy_of_dataset(data_tensor)
    attr_avg_info = get_avg_info_of_attribute(data_tensor, attribute_index)
    info_gain = dataset_ent - attr_avg_info
    return info_gain

def get_selected_attribute(data_tensor):
    num_attrs = data_tensor.shape[1] - 1
    attr_info_gains = {}
    
    for attr_index in range(num_attrs):
        info_gain = get_information_gain(data_tensor, attr_index)
        attr_info_gains[attr_index] = info_gain
    
    selected_attr_index = max(attr_info_gains, key=attr_info_gains.get)
    
    return attr_info_gains, selected_attr_index
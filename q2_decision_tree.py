import pandas as pd
import numpy as np
import os

# TO DO: Import any additional packages



# TO DO: implement the entropy calculation for a given Series.
def entropy(S):
    if len(S) == 0:  
        return 0
    total = sum(S)
    impurity = 0
    for count in S:
        if count == 0:  
            continue
        probability = count / total
        impurity -= probability * np.log2(probability)
    return impurity



# To Do: implement miscalculation loss for a given Series.
def misclassification(S):
    if len(S) == 0:
        return 0
    total = sum(S)
    max_probability = max(count / total for count in S)
    return 1 - max_probability




def gini(S):
    total = 0
    for value in S:
        total += value
    impurity = 1
    for num in S:
        impurity -= (num / total) **2
    return impurity


# To Do: Implement the information gain that takes a set of splits using
#        a single feature and given loss criterion to provide the 
#        information gain.

def calculate_gain(subset_splits, dataset, label, criterion):

    label_counts = dataset[label].value_counts()
    initial_impurity = criterion(label_counts)
    
    
    weighted_impurity = 0
    total_samples = sum(sum(counts) for counts in subset_splits.values())
    
    for counts in subset_splits.values():
       
        subset_impurity = criterion(counts)
       
        weight = sum(counts) / total_samples
        weighted_impurity += weight * subset_impurity
    
 
    info_gain = initial_impurity - weighted_impurity
    return info_gain



# To Do: Implement a function that splits a dataset by a specific
#        feature.
def dataset_split_by_feature(dataset, feature, label):
    subset = {}
    for i, value in enumerate(dataset[feature]):
            classification = dataset[label][i]
            if value not in subset:
                subset[value] = [0, 0]
            if classification == "e":
                subset[value][0] += 1
            else:
                subset[value][1] += 1
    return subset        

    



# To Do: Fill find_best_split(dataset, label, criterion) function
#        which will find the best split for a given dataset, label,
#        and criterion. It should out using the given string output
#        format.

def find_best_split(dataset, label, criterion="gini"):
    criterion_func = {"gini": gini, "entropy": entropy, "misclassification": misclassification}.get(criterion, gini)
    best_feature = None
    best_gain = -np.inf
    for feature in dataset.columns:
        if feature == label:
            continue
        split = dataset_split_by_feature(dataset, feature, label)
        gain = calculate_gain(split, dataset, label, criterion_func)  
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    print(best_feature, best_gain)
    return best_feature, best_gain

def find_best_split2(dataset, label, criterion="gini"):
    
    for feature in dataset.columns:
        if feature != label:
            split = dataset_split_by_feature(dataset, feature, label)
            print(feature, split)
            print(calculate_gain(split, dataset, label, criterion))
            
            

    best_feature = best_gain = 0
    return best_feature, best_gain


# If you used the starter code, you won't have to change any of the below
# code except for filling in your name and adjusting the criterion if you
# are in CS4361 and choose not to implement additional criterion.
if __name__ == '__main__':
  # read file
  name = "Cartwright_Brandon"
  os.makedirs(name=name)
  data = pd.read_csv("agaricus-lepiota.csv")
  label = "class"
  
  # Find the best feature using suggested methods
  criteria = ["gini", "entropy", "misclassification"]
  f = open(f"{name}/mushrooms.txt", "w")
  for criterion in criteria:
    best_feature, best_gain = find_best_split(data, label, criterion)
    f.write(f"Best feature: {best_feature} Using: {criterion}-Gain: {best_gain:.2f}\n")
  f.close()
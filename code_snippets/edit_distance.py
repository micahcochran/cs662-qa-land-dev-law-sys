""" Could be used as replacement for levenshtein distance from nltk.metrics.distance
    Memoized version of the Levenshtein distance function with spelling adjustment
"""
import os
import numpy as np
import collections
import ast

file_path = f'{os.getcwd()}'

def min_edit_distance_adjusted(token1, token2, confusion_data):
    counts = get_sub_counts(token1, token2)
    distance_matrix = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distance_matrix[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distance_matrix[0][t2] = t2
        
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            sub_cost = 0
            del_cost = 1
            ins_cost = 1
            if (token1[t1-1] == token2[t2-1]):
                sub_cost = 0
            else:
                sub_cost = generate_spelling_adjusted_sub_cost(token1[t1-1]+token2[t2-1], counts, confusion_data)
            
            distance_matrix[t1][t2] = min(distance_matrix[t1 - 1][t2 - 1] + sub_cost, distance_matrix[t1 - 1][t2] + del_cost, 
                                    distance_matrix[t1][t2 - 1] + ins_cost)

    # print_distance_matrix(distance_matrix, len(token1), len(token2))
    return distance_matrix[len(token1)][len(token2)]

def get_sub_counts(token1, token2):
    counts = []
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] != token2[t2-1]):
                counts.append(token1[t1-1])
    counter=collections.Counter(counts)
    return dict(counter)
    
def generate_spelling_adjusted_sub_cost(token1, counts, adjustment_dict):
    N = counts[token1[0]]
    x = adjustment_dict[token1]
    if x == 0:
        return 0
    sub_cost = N/x
    return sub_cost

def print_distance_matrix(distance_matrix, length1, length2):
    for t1 in range(length1 + 1):
        for t2 in range(length2 + 1):
            print(int(distance_matrix[t1][t2]), end=" ")
        print()
        
with open(f'{file_path}/subconfusion.data') as f:
    data = f.read()
    
confusion_data = ast.literal_eval(data) # * pass this to correct common spelling errors

print("Minimum Edit Distance Between", "intention and execution:", min_edit_distance_adjusted("intention", "execution", confusion_data))
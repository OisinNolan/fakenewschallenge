from csv import reader, writer
from tqdm import tqdm
from functools import reduce
import math
import numpy as np

with open('./data/train_stances.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    next(csv_reader, None) # skip header

    stance_dict = {}
    row_array = []

    # Use headline as dict key and store associated rows in the csv file
    # Then we know which rows to put in which dataset based on their stances
    # This means we can just split the set of stances up (the set isstance_dict.keys())
    #   and then send the relevant rows along to the two new csv files.
    for ii, row in enumerate(tqdm(csv_reader)):
        if stance_dict.get(row[0]):
            stance_dict[row[0]].append(ii)
        else:
            stance_dict[row[0]] = [ii]
        row_array.append(row)

    # Set the first 0.7 headlines as the train headlines, and get the associated rows
    train_cutoff = math.floor(len(stance_dict) * 0.7)    
    train_row_ids = reduce(
        lambda flat_list, sub_list: flat_list + sub_list,
        list(stance_dict.values())[:train_cutoff]
    )
    train_rows = [row_array[i] for i in train_row_ids]

    # The row ids _not_ in train_row_ids must be the validation rows
    val_rows = [row for i, row in enumerate(row_array) if i not in train_row_ids]

    # Make sure there aren't any shared headlines across the two sets
    assert (not any(np.isin([row[0] for row in train_rows], [row[0] for row in val_rows])))


    with open("./data/train_stances.newsplit.csv", "w") as f:
        writer = writer(f)
        for row in train_rows:
            writer.writerow(row)

    with open("./data/val_stances.newsplit.csv", "w") as f:
        writer = writer(f)
        for row in val_rows:
            writer.writerow(row)
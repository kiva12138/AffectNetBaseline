import pandas as pd
import json

if __name__ == '__main__':
    manually_annotated_file_list_train_path = r'/home/sh/AffectNet/training_classify_cleaned.csv'
    classify_need_columns = ['subDirectory_filePath', 'expression']
    
    nums = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0 ,5: 0, 6: 0, 7: 0, -1: 0}
    train_data_raw = pd.DataFrame(pd.read_csv(filepath_or_buffer=manually_annotated_file_list_train_path, usecols=classify_need_columns))
    for index, row in train_data_raw.iterrows():
        nums[row[classify_need_columns[1]]] += 1
        nums[-1] += 1
    nums = json.dumps(nums)
    print(nums)
    with open(r'/home/sh/AffectNet/train_mount.json', 'w') as outfile:
        json.dump(nums, outfile)
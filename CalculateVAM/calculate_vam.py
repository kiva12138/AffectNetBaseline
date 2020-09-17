import pandas as pd
import json

if __name__ == '__main__':
    annotated_file_list_path = r'/home/sh/AffectNet/training_classify_cleaned.csv'
    classify_need_columns = ['expression', 'valence', 'arousal']
    train_data_raw = pd.DataFrame(pd.read_csv(filepath_or_buffer=annotated_file_list_path, usecols=classify_need_columns))

    VAM = {'valence': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0 ,5: 0, 6: 0, 7: 0, -1: 0},
            'arousal': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0 ,5: 0, 6: 0, 7: 0, -1: 0}}
    va_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0 ,5: 0, 6: 0, 7: 0, -1: 0}
    
    for index, row in train_data_raw.iterrows():
        if row[classify_need_columns[1]] <= -2 or row[classify_need_columns[1]] >= 2 \
            or row[classify_need_columns[2]] <= -2 or row[classify_need_columns[2]] >= 2:
            continue

        VAM[classify_need_columns[1]][row[classify_need_columns[0]]] += row[classify_need_columns[1]]
        VAM[classify_need_columns[2]][row[classify_need_columns[0]]] += row[classify_need_columns[2]]

        VAM[classify_need_columns[1]][-1] += row[classify_need_columns[1]]
        VAM[classify_need_columns[2]][-1] += row[classify_need_columns[2]]

        va_count[row[classify_need_columns[0]]] += 1
        va_count[-1] += 1

    print('All:')
    print(VAM['valence'])
    print(VAM['arousal'])
    print(va_count, end='\n')

    for i in range(8):
        VAM[classify_need_columns[1]][i] /= va_count[i]
        VAM[classify_need_columns[2]][i] /= va_count[i]
    
    VAM[classify_need_columns[1]][-1] /= va_count[-1]
    VAM[classify_need_columns[2]][-1] /= va_count[-1]

    print('Mean:')
    print(VAM['valence'])
    print(VAM['arousal'])

    VAM = json.dumps(VAM)
    with open(r'/home/sh/AffectNet/vam.json', 'w') as outfile:
        json.dump(VAM, outfile)

    with open(r'./vam.json', 'w') as outfile2:
        json.dump(VAM, outfile2)

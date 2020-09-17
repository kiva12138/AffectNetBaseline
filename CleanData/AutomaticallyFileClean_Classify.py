import pandas as pd

if __name__ == '__main__':
    annotated_file_list_path = r'/home/sh/AffectNet/Automatically_Annotated/automatically_annotated.csv'
    classify_need_columns = ['subDirectory_filePath', 'expression']
    csv_columns = ['subDirectory_filePath',	'face_x', 'face_y',	'face_width', 'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal']

    train_data_raw = pd.DataFrame(pd.read_csv(filepath_or_buffer=annotated_file_list_path))
    train_data_new = pd.DataFrame(columns=(csv_columns))
    for index, row in train_data_raw.iterrows():
        if row[classify_need_columns[1]] > 7:
            continue
        train_data_new = train_data_new.append([dict(row)], ignore_index=True)
    train_data_new.to_csv('/home/sh/AffectNet/Automatically_Annotated/automatically_annotated_class_cleaned.csv')
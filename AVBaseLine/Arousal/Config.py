# manually_annotated_file_list_train_path = r'D:\pythontest\DataSet\AffectNet\ExtractedFiles\Manually_Annotated_file_lists\training_regression_cleaned.csv'
# manually_annotated_file_list_validation_path = r'D:\pythontest\DataSet\AffectNet\ExtractedFiles\Manually_Annotated_file_lists\valid_regression_cleaned.csv'
# manually_annotated_file_prefix = r'D:\pythontest\DataSet\AffectNet\ExtractedFiles\Manually_Annotated'

manually_annotated_file_list_train_path = r'/home/sh/AffectNet/training_regression_cleaned.csv'
manually_annotated_file_list_validation_path = r'/home/sh/AffectNet/validation_regression_cleaned.csv'
manually_annotated_file_prefix = r'/home/sh/AffectNet/Manually_Annotated/Manually_Annotated_Images'

# weights_save_path = r'./NetworkWithPytorch/AlexNet/Valence/weights/'
# log_dir = r'./NetworkWithPytorch/AlexNet/Valence/logdir/'
weights_save_path = r'/home/sh/VSRemote/NetworkWithPytorch/AlexNet/Arousal/weights/net.pth'
log_dir = r'./logdir/'

expressions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'Non-Face']
csv_columns = ['subDirectory_filePath',	'face_x', 'face_y',	'face_width', 'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal']
classify_need_columns = ['subDirectory_filePath', 'expression']
valence_need_columns = ['subDirectory_filePath', 'valence']
arousal_need_columns = ['subDirectory_filePath', 'arousal']


image_channels = 3
image_resize_height = 256
image_resize_width = 256
image_crop_height = 227
image_crop_width = 227
image_crop_pos_height = image_resize_height - image_crop_height
image_crop_pos_width = image_resize_width - image_crop_width
input_shape = (None, image_channels, image_crop_height, image_crop_width)
num_classes = 8

BATCH_SIZE = 256
EPOCHS = 20
lr = 0.001
momentum = 0.9
check_iter = 100
patience = 50

down_sample_max_number = 15000


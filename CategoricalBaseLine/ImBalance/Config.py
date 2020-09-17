manually_annotated_file_list_train_path = r'/home/sh/AffectNet/training_classify_cleaned.csv'
manually_annotated_file_list_validation_path = r'/home/sh/AffectNet/validation_classify_cleaned.csv'
manually_annotated_file_prefix = r'/home/sh/AffectNet/Manually_Annotated/Manually_Annotated_Images'

weights_save_path = r'/home/sh/VSRemote/NetworkWithPytorch/AffectNet/CategoricalBaseLine/ImBalance/weights/net.pth'
log_dir = r'./logdir/'

expressions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'Non-Face']
classify_need_columns = ['subDirectory_filePath', 'expression']


image_channels = 3
image_resize_height = 256
image_resize_width = 256
image_crop_height = 224
image_crop_width = 224
num_classes = 8

BATCH_SIZE = 256
EPOCHS = 30
lr = 0.01
lr_decrease_rate = 0.1
lr_decrease_iter = 10000
momentum = 0.9

check_iter = 100

continue_train = True
continue_epoch = 4
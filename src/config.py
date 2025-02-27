num_classes = 7
batch_size  = 64
num_epochs  = 75
img_size    = (224, 224, 3)


### Configuração do pipeline

use_graphic_preprocessing  = True
use_data_preprocessing     = True
use_data_augmentation      = True
use_class_weight           = False
use_fine_tuning            = True
use_cnn_as_classifier      = True
use_fine_tuning_at_layer   = 0



num_kfolds                 = 0
num_pca_components         = 100
cnn_model                  = 'VGG19'
classical_classifier_model = 'RandomForest'

num_classes = 7
batch_size  = 7
num_epochs  = 75
img_size    = (299, 299, 3)


### Configuração do pipeline

use_graphic_preprocessing  = True
use_data_preprocessing     = False
use_fine_tuning            = True
use_cnn_as_classifier      = False
use_fine_tuning_at_layer   = 0
visualize                  = False



num_kfolds                 = 10
num_pca_components         = None
cnn_model                  = 'Inception'
classical_classifier_model = 'RandomForest'

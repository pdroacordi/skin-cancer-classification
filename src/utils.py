import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.utils import to_categorical
import cv2

from skincancer.src.config import img_size, num_classes

def get_training_augmentor():
    """
    Retorna o pipeline de data augmentation para carregamento das imagens no treinamento.
    """
    import albumentations as a
    return a.Compose([
        a.Affine(
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            rotate=(-2, 2),
            scale=(0.98, 1.02),
            shear=0,
            p=0.3
        ),
        a.HorizontalFlip(p=0.2),
        a.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
    ])

def process_image(image, model_name, augment=False, augmentor=None,
                  segmentation=False, visualize=False):
    """
    Processa uma imagem aplicando:
      - Pré-processamento gráfico (remoção de pelos, contraste, etc.);
      - Data augmentation (se ativado e se o pipeline for fornecido);
      - Conversão para RGB, resize e pré-processamento específico do modelo.
    """
    original = image.copy()

    image = resize_image(image, img_size[:2])

    image = apply_graphic_preprocessing(
        image,
        segmentation=segmentation
    )

    if augment and augmentor is not None:
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         augmented = augmentor(image=image)
         image = augmented["image"]
    else:
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Antes')
        plt.axis('off')


        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.title('Depois')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    if model_name == "VGG19":
         image = preprocess_input_vgg19(image)
    elif model_name == "Inception":
         image = preprocess_input_inception(image)
    elif model_name == "ResNet":
         image = preprocess_input_resnet(image)
    elif model_name == "Xception":
         image = preprocess_input_xception(image)
    else:
         raise ValueError("Modelo não suportado.")

    return image

def load_paths_labels(file_path):
    paths, labels = [], []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                path, label = line.strip().split('\t')
                paths.append(path)
                labels.append(int(label))
            except ValueError:
                print(f"Line {idx + 1} is malformed: {line}")
    return np.array(paths), np.array(labels)


def apply_segmentation(image):
    import skincancer.src.preprocessing.segmentation as segmentation

    init_contour = segmentation.generate_adaptive_circular_contour(image, num_points=100)

    final_contour = segmentation.gvf_based_segmentation(
        image,
        init_contour,
        mu=0.05,
        iterations=400,
        delta_t=0.1,
        alpha=0.05,
        beta=0.01,
        gamma=1,
        kappa=10,
        iterations_snake=500
    )

    final_contour_int = final_contour.reshape((-1, 1, 2)).astype(np.int32)

    segmented_image = image.copy()
    cv2.drawContours(segmented_image, [final_contour_int], contourIdx=-1,
                     color=(0, 0, 255), thickness=2)

    return segmented_image

def apply_graphic_preprocessing(image,
                                segmentation=False):
    processed_image = image.copy()
    if segmentation:
        processed_image = apply_segmentation(processed_image)

    return processed_image


def resize_image(image, target_size):
    """
    Redimensiona a imagem para o tamanho especificado.

    Args:
        image (numpy.array): Imagem lida, em formato BGR ou RGB.
        target_size (tuple): Tamanho desejado no formato (altura, largura).

    Returns:
        numpy.array: Imagem redimensionada.
    """
    resized_image = cv2.resize(image, (target_size[1], target_size[0]))
    return resized_image

def load_and_preprocess_images(paths, model_name, labels=None, augment=False,
                               segmentation=False, visualize=False):
    """
        Carrega e pré-processa as imagens a partir dos caminhos fornecidos.
        Se 'augment' for True, aplica o pipeline de augmentation do treinamento.
    """
    images = []
    augmentor = get_training_augmentor() if augment else None
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Erro ao carregar a imagem: {path}")
        processed = process_image(image, model_name, augment=augment, augmentor=augmentor,
                                  segmentation=segmentation, visualize=visualize)
        images.append(processed)
    images = np.array(images)
    if labels is not None:
        labels = to_categorical(labels, num_classes)
        return images, labels
    return images

def data_generator(paths, labels,
                   batch_size=32,
                   model_name="VGG19",
                   segmentation=False,
                   shuffle=True,
                   augment=False,
                   visualize=False):
    """
    Gerador que carrega as imagens em batches, aplicando pré-processamento e (opcionalmente) data augmentation
    usando um pipeline próprio para o gerador.
    """
    augmentor = get_training_augmentor() if augment else None
    n = len(paths)
    idxs = np.arange(n)
    while True:
         if shuffle:
              np.random.shuffle(idxs)
         for start in range(0, n, batch_size):
              end = min(start + batch_size, n)
              batch_idxs = idxs[start:end]
              batch_images = []
              batch_labels = []
              for i in batch_idxs:
                   image_path = paths[i]
                   image = cv2.imread(image_path)
                   if image is None:
                        print(f"Falha ao ler a imagem: {image_path}")
                        continue
                   processed = process_image(image, model_name, augment=augment, augmentor=augmentor,
                                               segmentation=segmentation, visualize=visualize)
                   batch_images.append(processed)
                   batch_labels.append(labels[i])
              if len(batch_images) == 0:
                   continue
              batch_images = np.array(batch_images, dtype=np.float32)
              batch_labels = to_categorical(batch_labels, num_classes)
              yield batch_images, batch_labels
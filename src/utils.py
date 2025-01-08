import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt

from skincancer.src.config import img_size, num_classes


# Função para carregar caminhos e rótulos
def load_paths_labels(file_path):
    paths, labels = [], []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                path, label = line.strip().split('\t')
                paths.append(path)
                labels.append(int(label))
            except ValueError:
                print(f"Line {idx+1} is malformed: {line}")
    return np.array(paths), np.array(labels)

def remove_hairs(image_path, visualize=False):
    """
    Remove pelos de uma imagem dermatológica.

    Args:
        image_path (str): Caminho para a imagem.
        visualize (bool): Se True, exibe a imagem antes e depois do processamento.

    Returns:
        cleaned_image (numpy.ndarray): Imagem com os pelos removidos.
    """
    # Ler a imagem
    print("Removendo pelos...")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Aplicar filtro morfológico para destacar os pelos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))  # Kernel para top-hat
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    #Criar uma máscara binária para os pelos
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    #Preencher as regiões dos pelos com inpainting
    cleaned_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    #Visualizar resultados
    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 3, 1)
        plt.title("Imagem Original")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        plt.title("Máscara dos Pelos")
        plt.imshow(mask, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Imagem com Pelos Removidos")
        plt.imshow(cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB))

        plt.show()

    return cleaned_image

def extract_roi(image, visualize=False):
    """
    Extrai a Região de Interesse (ROI) em escala de cinza e aplica pré-processamentos gráficos.

    Args:
        image (numpy.ndarray): Imagem.
        visualize (bool): Se True, exibe a imagem original e os resultados intermediários.
    Returns:
        roi (numpy.ndarray): Imagem da região de interesse extraída e pré-processada (em RGB).
    """
    print("Extraindo ROI...")
    # Converter para escala de cinza apenas para processamento
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remoção de ruído com filtro Gaussiano
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Limiarização para binarizar a imagem
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar contornos e extrair a ROI
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Se nenhum contorno for encontrado, retornar a imagem original redimensionada
        roi = cv2.resize(image, img_size)
        return roi

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Recortar a ROI da imagem original (em colorido)
    roi = image[y:y + h, x:x + w]

    # Visualizar resultados
    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.title("Imagem Original")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 2, 2)
        plt.title("ROI Extraída (RGB)")
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        plt.show()

    return roi  # Retorna a ROI em RGB


def custom_data_generator(
        paths, labels,
        batch_size=32,
        model_name="VGG19",
        use_graphic_preprocessing=False,
        use_hair_removal=True,
        shuffle=True,
        augment=False
):
    """
    Gera batches de (X, y) usando pré-processamento manual + data augmentation.
    - `paths`: array de caminhos (strings) para as imagens.
    - `labels`: array de rótulos (int) do mesmo tamanho de `paths`.
    - `batch_size`: tamanho do batch.
    - `model_name`: para decidir qual preprocess_input usar (VGG19, Inception, etc.).
    - `use_graphic_preprocessing`: se True, faz remove_hairs e extract_roi.
    - `augment`: se True, aplica data augmentation com albumentations.

    Retorna um generator infinito (yield) que pode ser usado em model.fit().
    """
    n = len(paths)
    idxs = np.arange(n)  # índice das imagens

    while True:
        if shuffle:
            np.random.shuffle(idxs)  # embaralha a cada época

        # Percorre o dataset em steps de batch
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idxs = idxs[start:end]

            batch_images = []
            batch_labels = labels[batch_idxs]

            for i in batch_idxs:
                image_path = paths[i]

                # 1) Ler a imagem
                image = cv2.imread(image_path)  # BGR
                if image is None:
                    # se der erro de leitura
                    continue

                # 2) Se use_hair_removal, chamar remove_hairs
                if use_graphic_preprocessing and use_hair_removal:
                    image = remove_hairs(image_path, visualize=False)
                else:
                    pass

                # 3) Se use_graphic_preprocessing, extrair ROI
                if use_graphic_preprocessing:
                    image = extract_roi(image, visualize=False)

                # 4) Data augmentation (caso augment=True)
                if augment:
                    import albumentations as A
                    augmentor = A.Compose([
                        A.RandomRotate90(p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.1),
                        A.RandomBrightnessContrast(p=0.2)
                    ])

                    # albumentations trabalha em RGB
                    # mas o cv2.imread é BGR. Converta antes:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Aplique as transformações
                    augmented = augmentor(image=image)
                    image = augmented["image"]
                else:
                    # Se não for augment, só converter BGR->RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 5) Resize
                image = cv2.resize(image, img_size)

                # 6) Chamar a preprocess_input correspondente
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

                batch_images.append(image)

            batch_images = np.array(batch_images, dtype=np.float32)

            # 7) Converter labels para one-hot
            batch_labels = to_categorical(batch_labels, num_classes)

            yield batch_images, batch_labels

# Pré-processamento gráfico (ROI, ruído, HSV, resize)
def preprocess_image(path, model_name, use_graphic_preprocessing=False, use_hair_removal=True, visualize=False):
    if use_graphic_preprocessing:
        if use_hair_removal:
            image = remove_hairs(path, visualize=visualize)
        else:
            image = cv2.imread(path)
        roi = extract_roi(image, visualize=visualize)
    else:
        roi = cv2.imread(path)

    # Redimensionar para o tamanho esperado pela CNN
    roi_resized = cv2.resize(roi, img_size)
    # Normalizar para o formato esperado pela CNN
    if model_name == "VGG19":
        roi_resized = preprocess_input_vgg19(roi_resized)
    elif model_name == "Inception":
        roi_resized = preprocess_input_inception(roi_resized)
    elif model_name == "ResNet":
        roi_resized = preprocess_input_resnet(roi_resized)
    elif model_name == "Xception":
        roi_resized = preprocess_input_xception(roi_resized)
    else:
        raise ValueError(
            f"Modelo {model_name} não suportado. Escolha entre 'VGG19', 'Inception', 'ResNet' ou 'Xception'.")

    return roi_resized

# Carregar e pré-processar imagens
def load_and_preprocess_images(paths, model_name, labels=None, use_graphic_preprocessing=False, cache=None):
    images = []
    for path in paths:
        if cache is not None and path in cache:
            image = cache[path]
        else:
            image = preprocess_image(path, model_name, use_graphic_preprocessing)
            if cache is not None:
                cache[path] = image
        images.append(image)
    images = np.array(images)
    if labels is not None:
        labels = to_categorical(labels, num_classes)
        return images, labels
    else:
        return images

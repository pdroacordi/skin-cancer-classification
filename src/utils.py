import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt

# Parâmetros
num_classes = 7
img_size = (224, 224)
batch_size = 64
n_epochs = 10  # Para experimentos iniciais, use menos épocas para acelerar.
train_files_path = "../res/train_files.txt"
val_files_path = "../res/val_files.txt"

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

# Pré-processamento gráfico (ROI, ruído, HSV, resize)
def preprocess_image(path, model_name, use_graphic_preprocessing=False, use_hair_removal=True, visualize=False):
    if use_hair_removal:
        image = remove_hairs(path, visualize=visualize)
    else:
        image = cv2.imread(path)

    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if use_graphic_preprocessing:
        roi = extract_roi(image, visualize=visualize)
    else:
        roi = image

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
def load_and_preprocess_images(paths, model_name, labels, use_graphic_preprocessing=False):
    images = [preprocess_image(path, model_name, use_graphic_preprocessing) for path in paths]
    return np.array(images), to_categorical(labels, num_classes)
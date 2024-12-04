import numpy as np
from keras.src.applications.inception_v3 import InceptionV3
from keras.src.applications.resnet import ResNet50
from keras.src.applications.xception import Xception
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt

# Parâmetros
num_classes = 6
img_size = (224, 224)
batch_size = 64
n_epochs = 10  # Para experimentos iniciais, use menos épocas para acelerar.
k_folds = 5
train_files_path = "../res/train_files.txt"
val_files_path = "../res/val_files.txt"

# Função para carregar caminhos e rótulos
def load_paths_labels(file_path):
    paths, labels = [], []
    with open(file_path, 'r') as f:
        for line in f:
            path, label = line.strip().split(',')
            paths.append(path)
            labels.append(int(label))
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
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Passo 1: Aplicar filtro morfológico para destacar os pelos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))  # Kernel para top-hat
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Passo 2: Criar uma máscara binária para os pelos
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Passo 3: Preencher as regiões dos pelos com inpainting
    cleaned_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Visualizar resultados
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
        roi (numpy.ndarray): Imagem da região de interesse extraída e pré-processada (em grayscale).
    """

    #Remoção de ruído com filtro Gaussiano
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    #Limiarização para binarizar a imagem
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #Encontrar contornos e extrair a ROI
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    roi = image[y:y + h, x:x + w]  # Recortar a ROI da imagem original

    # Visualizar resultados
    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 2, 1)
        plt.title("Imagem Original")
        plt.imshow(image, cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title("ROI Pré-processada (Grayscale)")
        plt.imshow(roi, cmap="gray")


        plt.show()

    return roi  # Retorna a ROI em escala de cinza

# Pré-processamento gráfico (ROI, ruído, HSV, resize)
def preprocess_image(path, use_graphic_preprocessing=False, use_hair_removal=True):
    if use_hair_removal:
        image = remove_hairs(path, visualize=False)
    else:
        image = cv2.imread(path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if use_graphic_preprocessing:
        roi = extract_roi(gray_image, visualize=False)
    else:
        roi = gray_image

    # Redimensionar para o tamanho esperado pela CNN
    roi_resized = cv2.resize(roi, img_size)
    # Normalizar para o formato esperado pela CNN
    roi_resized = preprocess_input(roi_resized)
    return roi_resized

# Carregar e pré-processar imagens
def load_and_preprocess_images(paths, labels, use_graphic_preprocessing=False):
    images = [preprocess_image(path, use_graphic_preprocessing) for path in paths]
    return np.array(images), to_categorical(labels, num_classes)

# Pré-processamento de dados com PCA
def apply_pca(features, n_components=100):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)

# Pré-processamento de dados com batch normalization
def apply_batch_normalization(features):
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

# Pré-processamento de dados
def pre_process_data(features, n_components=100):
    features = apply_pca(features, n_components=n_components)
    features = apply_batch_normalization(features)
    return features

def load_cnn_model(model_name, input_shape=(224, 224, 3)):
    """
    Carrega um modelo CNN pré-treinado.

    Args:
        model_name (str): Nome do modelo ('VGG19', 'Inception', 'ResNet', 'Xception').
        input_shape (tuple): Tamanho de entrada das imagens.

    Returns:
        model (keras.Model): Modelo pré-treinado configurado para extração de características.
    """
    if model_name == "VGG19":
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "Inception":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "ResNet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "Xception":
        base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Modelo {model_name} não suportado. Escolha entre 'VGG19', 'Inception', 'ResNet' ou 'Xception'.")

    # Conectar a camada de pooling global para extração de características
    model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))
    return model


def extract_features_with_cnn(paths, model_name="VGG19", use_graphic_preprocessing=False):
    """
    Extrai características de imagens usando um modelo CNN pré-treinado.

    Args:
        paths (list): Lista de caminhos das imagens.
        model_name (str): Nome do modelo ('VGG19', 'Inception', 'ResNet', 'Xception').
        use_graphic_preprocessing (bool): Se True, aplica pré-processamento gráfico.

    Returns:
        features (numpy.ndarray): Vetores de características extraídos.
    """
    # Carregar o modelo CNN
    model = load_cnn_model(model_name, input_shape=(224, 224, 3))

    features = []
    for path in paths:
        img = preprocess_image(path, use_graphic_preprocessing)
        features.append(model.predict(np.expand_dims(img, axis=0)).flatten())
    return np.array(features)

# Classificadores clássicos
def train_classical_classifier(features, labels, classifier_type="RandomForest"):
    if classifier_type == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    elif classifier_type == "SVM":
        model = SVC(probability=True, random_state=42)
    elif classifier_type == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif classifier_type == "KNN+SVM":
        knn = KNeighborsClassifier(n_neighbors=5)
        svm = SVC(probability=True, random_state=42)
        model = VotingClassifier(estimators=[
            ('knn', knn),
            ('svm', svm)
        ], voting='soft') #'soft' usa as probabilidades para combinação
    model.fit(features, labels)
    return model

# Treinamento e comparação de cenários
def run_experiment(train_paths, train_labels, val_paths, val_labels,
                   model_name="VGG19", use_graphic_preprocessing=False,
                   use_data_preprocessing=False, use_cnn_classifier=True,
                   classical_classifier="RandomForest"):

    if use_cnn_classifier:
        print(f"\nCenário - Modelo: {model_name}, Pré-Processamento Gráfico: {use_graphic_preprocessing}, Pré-Processamento de Dados: {use_data_preprocessing}")
    else:
        print(f"\nCenário - Classificador: {classical_classifier}, Pré-Processamento Gráfico: {use_graphic_preprocessing}, Pré-Processamento de Dados: {use_data_preprocessing}")

    if use_cnn_classifier:
        # CNN como classificador
        x_train, y_train = load_and_preprocess_images(train_paths, train_labels, use_graphic_preprocessing)
        x_val, y_val = load_and_preprocess_images(val_paths, val_labels, use_graphic_preprocessing)

        # Construir modelo CNN
        base_model = load_cnn_model(model_name, input_shape=(224, 224, 3))
        base_model.trainable = False
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=n_epochs, batch_size=batch_size, verbose=1)
        model.save(f"../models/{model_name}_classifier.h5")
        val_predictions = np.argmax(model.predict(x_val), axis=1)
    else:
        # CNN como extratora de características
        train_features = extract_features_with_cnn(train_paths, model_name=model_name, use_graphic_preprocessing=use_graphic_preprocessing)
        val_features = extract_features_with_cnn(val_paths, model_name=model_name, use_graphic_preprocessing=use_graphic_preprocessing)

        # Aplicar pré-processamento
        if use_data_preprocessing:
            train_features = pre_process_data(train_features)
            val_features = pre_process_data(val_features)

        # Classificador clássico
        clf = train_classical_classifier(train_features, train_labels, classical_classifier)
        val_predictions = clf.predict(val_features)

    # Relatório e matriz de confusão
    print("\nRelatório de Classificação:")
    print(classification_report(val_labels, val_predictions, digits=4))
    print("\nMatriz de Confusão:")
    print(confusion_matrix(val_labels, val_predictions))


train_paths, train_labels = load_paths_labels(train_files_path)
val_paths, val_labels = load_paths_labels(val_files_path)


# Executar experimentos
#run_experiment(train_paths, train_labels, val_paths, val_labels, use_graphic_preprocessing=True, use_data_preprocessing=True, use_cnn_classifier=True)
#run_experiment(train_paths, train_labels, val_paths, val_labels, use_graphic_preprocessing=False, use_data_preprocessing=False, use_cnn_classifier=False, classical_classifier="RandomForest")

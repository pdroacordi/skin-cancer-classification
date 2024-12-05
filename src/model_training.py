import os

import numpy as np
from tensorflow.keras.applications import VGG19, InceptionV3, ResNet50, Xception
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.backend import clear_session
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import *

# Parâmetros
num_classes = 7
img_size = (224, 224)
batch_size = 64
n_epochs = 5  # Para experimentos iniciais, use menos épocas para acelerar.
train_files_path = "../res/train_files.txt"
val_files_path = "../res/val_files.txt"

# Pré-processamento de dados com PCA
def apply_pca(train_features, val_features, save_path, n_components=100):
    pca = PCA(n_components=n_components)
    train_features_pca = pca.fit_transform(train_features)
    val_features_pca = pca.transform(val_features)
    from joblib import dump
    dump(pca, f"{save_path}/pca.joblib")
    return train_features_pca, val_features_pca

# Pré-processamento de dados com normalização
def apply_batch_normalization(train_features, val_features, save_path):
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    from joblib import dump
    dump(scaler, f"{save_path}/scaler.joblib")
    return train_features_scaled, val_features_scaled

# Pré-processamento de dados
def pre_process_data(train_features, val_features, save_path, n_components=100):
    train_features_scaled, val_features_scaled = apply_batch_normalization(train_features, val_features, save_path)
    train_features_pca, val_features_pca = apply_pca(train_features_scaled, val_features_scaled, save_path, n_components=n_components)
    return train_features_pca, val_features_pca


def load_or_train_cnn_model(model_name, use_cnn_classifier, input_shape=(224, 224, 3), fine_tune=False, fine_tune_at=0,
                   save_path="", x_train=None, y_train=None, x_val=None, y_val=None):
    """
    Carrega ou treina uma CNN pré-treinada e a salva conforme o uso (classificador ou extrator de características).

    Args:
        model_name (str): Nome do modelo ('VGG19', 'Inception', 'ResNet', 'Xception').
        use_cnn_classifier (bool): Define se o modelo terá a camada densamente conectada (True) ou não (False).
        input_shape (tuple): Tamanho de entrada das imagens.
        fine_tune (bool): Se True, realiza fine-tuning.
        fine_tune_at (int): Camada a partir da qual será realizado o fine-tuning.
        save_path (str): Caminho em que deve ser salvo o modelo.
        x_train (numpy.ndarray): Conjunto de dados de treino.
        y_train (numpy.ndarray): Conjunto de classes de treino.
        x_val (numpy.ndarray): Conjunto de dados de validação.
        y_val (numpy.ndarray): Conjunto de classes de validação.

    Returns:
        model (keras.Model): Modelo configurado e salvo.
        loaded (bool): Define se o modelo foi criado ou se foi carregado.
    """
    if os.path.exists(save_path):
        print(f"Carregando modelo existente: {save_path}")
        return load_model(save_path), True

    if model_name == "VGG19":
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
        fine_tune_at = fine_tune_at or 15
    elif model_name == "Inception":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        fine_tune_at = fine_tune_at or 40
    elif model_name == "ResNet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
        fine_tune_at = fine_tune_at or 25
    elif model_name == "Xception":
        base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        fine_tune_at = fine_tune_at or 30
    else:
        raise ValueError(
            f"Modelo {model_name} não suportado. Escolha entre 'VGG19', 'Inception', 'ResNet' ou 'Xception'.")

    if fine_tune:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False  # Congelar camadas iniciais
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True  # Descongelar camadas superiores
    else:
        base_model.trainable = False  # Congelar todas as camadas se não for fine-tuning

    if use_cnn_classifier:
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
    else:
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(256, activation='relu'),
            Dropout(0.5)
        ])

    if x_train is not None and y_train is not None:
        print("Treinando modelo...")
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=n_epochs, batch_size=batch_size, verbose=1)

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        print(f"Salvando modelo treinado em: {save_path}")
        model.save(save_path)

    return model, False


def extract_features_with_cnn(paths, model_name="VGG19", use_graphic_preprocessing=False,
                              fine_tune=False, fine_tune_at=0, save_path="", train_paths="",
                              train_labels="", val_paths="", val_labels=""):
    """
    Extrai características de imagens usando um modelo CNN pré-treinado.

    Args:
        model_name (str): Nome do modelo ('VGG19', 'Inception', 'ResNet', 'Xception').
        use_graphic_preprocessing (bool): Se True, aplica pré-processamento gráfico.
        fine_tune (bool): Se True, realiza fine-tuning.
        fine_tune_at (int): Camada a partir da qual será realizado o fine-tuning.
        save_path (str): Caminho em que deve ser salvo o modelo.
        train_paths (numpy.ndarray): Lista de imagens de treinamento
        train_labels (numpy.ndarray): Lista de classes de treinamento
        val_paths (numpy.ndarray): Lista de imagens de validação
        val_labels (numpy.ndarray): Lista de classes de validação
    Returns:
        features (numpy.ndarray): Vetores de características extraídos.
    """
    x_train, y_train = None, None  # Somente necessário para fine-tuning
    x_val, y_val = None, None  # Somente necessário para fine-tuning
    if fine_tune:
        x_train, y_train = load_and_preprocess_images(train_paths, model_name, train_labels, use_graphic_preprocessing)
        x_val, y_val = load_and_preprocess_images(val_paths, model_name, val_labels, use_graphic_preprocessing)

    model, loaded = load_or_train_cnn_model(
        model_name=model_name,
        use_cnn_classifier=False,
        input_shape=(224, 224, 3),
        fine_tune=fine_tune,
        fine_tune_at=fine_tune_at,
        save_path=save_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val
    )
    if not fine_tune:
        images = [preprocess_image(path, model_name, use_graphic_preprocessing) for path in paths]
        images = np.array(images)
    else:
        images = x_train

    # Extrair características
    features = model.predict(images)
    return features


# Classificadores clássicos
def train_classical_classifier(features, labels, classifier_type="RandomForest", save_model=False, save_path=""):
    if save_model and os.path.exists(save_path):
        from joblib import load
        print(f"Carregando modelo clássico existente: {save_path}")
        model = load(save_path)
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
    else:
        raise ValueError(
            f"Modelo {classifier_type} não suportado. Escolha entre 'RandomForest', 'SVM', 'KNN' ou 'KNN+SVM'.")
    model.fit(features, labels)

    if save_model:
        from joblib import dump
        dump(model, save_path)
        print(f"Salvando modelo treinado em: {save_path}")

    return model

# Treinamento e comparação de cenários
def run_experiment(all_paths, all_labels,
                   model_name="VGG19", use_graphic_preprocessing=False,
                   use_data_preprocessing=False, use_cnn_classifier=True,
                   classical_classifier="RandomForest", fine_tune=False, fine_tune_at=0,
                   k_folds=5):

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold = 1
    all_val_labels = []
    all_val_predictions = []

    for train_index, val_index in kf.split(all_paths):
        print(f"\nFold {fold}/{k_folds}")
        train_paths, val_paths = all_paths[train_index], all_paths[val_index]
        train_labels, val_labels = all_labels[train_index], all_labels[val_index]

        model_type = "classifier" if use_cnn_classifier else "extractor"
        save_path = f"../models/CNN/{model_name}/{model_name.lower()}_graphic_{use_graphic_preprocessing}_fine_tune_{fine_tune}{f'_fine_tune_at_{fine_tune_at}' if fine_tune_at != 0 else ''}_{model_type}.h5"

        if use_cnn_classifier:
            # CNN como classificador
            print(f"\nCenário - Modelo: {model_name}, Fine-Tune: {fine_tune}, {f"Camada: {fine_tune_at}, " if fine_tune_at != 0 else ""}Pré-Processamento Gráfico: {use_graphic_preprocessing}")
            x_train, y_train = load_and_preprocess_images(train_paths, model_name, train_labels, use_graphic_preprocessing)
            x_val, y_val = load_and_preprocess_images(val_paths, model_name, val_labels, use_graphic_preprocessing)

            # Construir modelo CNN
            model, loaded = load_or_train_cnn_model(
                model_name=model_name,
                fine_tune=fine_tune,
                fine_tune_at=fine_tune_at,
                save_path=save_path,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val
            )

            val_predictions = np.argmax(model.predict(x_val), axis=1)
            val_labels_non_categorical = np.argmax(y_val, axis=1)
        else:
            # CNN como extratora de características
            print(f"\nCenário - Classificador: {classical_classifier}, CNN: {model_name}, Fine-Tune: {fine_tune}, {f"Camada: {fine_tune_at}, " if fine_tune_at != 0 else ""}Pré-Processamento Gráfico: {use_graphic_preprocessing}, Pré-Processamento de Dados: {use_data_preprocessing}")

            clf_save_path = f"../models/CLASSIC/{classical_classifier}/{classical_classifier.lower()}_{model_name.lower()}_fine_tune_{fine_tune}{f"_fine_tune_at_{fine_tune_at}" if fine_tune_at != 0 else ""}_graphic_{use_graphic_preprocessing}/{model_type}.joblib"
            pre_save_path = f"../models/CLASSIC/{classical_classifier}/{classical_classifier.lower()}_{model_name.lower()}_fine_tune_{fine_tune}{f"_fine_tune_at_{fine_tune_at}" if fine_tune_at != 0 else ""}_graphic_{use_graphic_preprocessing}"

            train_features = extract_features_with_cnn(paths=train_paths, model_name=model_name, use_graphic_preprocessing=use_graphic_preprocessing,
                                                       fine_tune=fine_tune, fine_tune_at=fine_tune_at, save_path=save_path, train_paths=train_paths,
                                                       val_paths=val_paths, train_labels=train_labels, val_labels=val_labels)

            val_features = extract_features_with_cnn(paths=val_paths, model_name=model_name, use_graphic_preprocessing=use_graphic_preprocessing,
                                                     fine_tune=fine_tune, fine_tune_at=fine_tune_at, save_path=save_path, train_paths=train_paths,
                                                     val_paths=val_paths, train_labels=train_labels, val_labels=val_labels)

            if use_data_preprocessing:
                train_features, val_features = pre_process_data(train_features=train_features, val_features=val_features, save_path=pre_save_path)


            clf = train_classical_classifier(train_features, train_labels, classical_classifier, True, clf_save_path)

            val_predictions = clf.predict(val_features)
            val_labels_non_categorical = val_labels

        all_val_labels.extend(val_labels_non_categorical)
        all_val_predictions.extend(val_predictions)

        # Relatório e matriz de confusão para o fold atual
        print(f"\nRelatório de Classificação para o Fold {fold}:")
        print(classification_report(val_labels_non_categorical, val_predictions, digits=4))
        print(f"\nMatriz de Confusão para o Fold {fold}:")
        print(confusion_matrix(val_labels_non_categorical, val_predictions))
        clear_session()
        fold += 1

    print("\nRelatório de Classificação Geral:")
    print(classification_report(all_val_labels, all_val_predictions, digits=4))
    print("\nMatriz de Confusão Geral:")
    print(confusion_matrix(all_val_labels, all_val_predictions))



train_paths, train_labels = load_paths_labels(train_files_path)
val_paths, val_labels = load_paths_labels(val_files_path)

all_paths = np.concatenate((train_paths, val_paths))
all_labels = np.concatenate((train_labels, val_labels))

run_experiment( all_paths, all_labels, "VGG19", True, True, False, "KNN", True, 0, 3 )

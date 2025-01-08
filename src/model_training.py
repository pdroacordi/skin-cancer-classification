import math
import os

import numpy as np
import tensorflow
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras.applications import VGG19, InceptionV3, ResNet50, Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.python.keras.backend import clear_session

import skincancer.src.config
from utils import (
    load_paths_labels,
    load_and_preprocess_images,
    custom_data_generator
)

###############################################################################
# Funções de PCA / Scaler
###############################################################################
# Pré-processamento de dados com PCA
def apply_pca(train_features, val_features, save_path, n_components=100):
    pca = PCA(n_components=n_components)
    train_features_pca = pca.fit_transform(train_features)
    val_features_pca = pca.transform(val_features)
    from joblib import dump
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    dump(pca, f"{save_path}pca.joblib")
    return train_features_pca, val_features_pca

# Pré-processamento de dados com normalização
def apply_batch_normalization(train_features, val_features, save_path):
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    from joblib import dump
    dump(scaler, f"{save_path}scaler.joblib")
    return train_features_scaled, val_features_scaled

# Pré-processamento de dados
def pre_process_data(train_features, val_features, save_path, n_components=100):
    train_features_scaled, val_features_scaled = apply_batch_normalization(train_features, val_features, save_path)
    train_features_pca, val_features_pca = apply_pca(train_features_scaled, val_features_scaled, save_path, n_components=n_components)
    return train_features_pca, val_features_pca

###############################################################################
# Construção / Carregamento do Modelo CNN (sem treinar dentro da função)
###############################################################################
def load_or_build_cnn_model(model_name="VGG19",
                            input_shape=(224, 224, 3),
                            fine_tune=False,
                            fine_tune_at=0,
                            save_path="",
                            num_classes=7):
    """
    Cria ou carrega um modelo CNN e o salva em 'save_path'.
    Se já existir 'save_path', o modelo é carregado.

    Args:
        model_name (str): Nome do modelo ('VGG19', 'Inception', 'ResNet', 'Xception').
        input_shape (tuple): Tamanho de entrada das imagens.
        fine_tune (bool): Se True, realiza fine-tuning.
        fine_tune_at (int): Camada a partir da qual será realizado o fine-tuning.
        save_path (str): Caminho em que deve ser salvo o modelo.
        num_classes (int): Número de classes de treino.

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

    x = layers.GlobalAveragePooling2D(name='gap')(base_model.output)
    x = layers.Dense(256, activation='relu', name='dense_features')(x)
    x = layers.Dropout(0.5, name='dropout_features')(x)
    predictions = layers.Dense(num_classes, activation='softmax', name='final_predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    print(f"[CNN] Salvando modelo (recém construído) em {save_path}")
    model.save(save_path)

    return model, False

###############################################################################
# K-Fold CNN
###############################################################################
def run_kfold_cnn(
        all_paths,
        all_labels,
        model_name="VGG19",
        fine_tune=False,
        fine_tune_at=0,
        k_folds=5,
        epochs=10,
        batch_size=32,
        use_graphic_preprocessing=False,
        save_path=""
):
    """
    Exemplo de K-Fold end-to-end na CNN.
    Em cada fold, treina uma CNN do zero (ou com fine-tune).
    CUIDADO: Alto custo computacional se k_folds > 1, pois treinará
             a CNN repetidas vezes.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold = 1

    all_preds = []
    all_trues = []

    for train_idx, val_idx in kf.split(all_paths):
        print(f"\n[CNN K-Fold] Fold {fold}/{k_folds}")
        train_paths_fold  = all_paths[train_idx]
        val_paths_fold    = all_paths[val_idx]
        train_labels_fold = all_labels[train_idx]
        val_labels_fold   = all_labels[val_idx]

        # Cria geradores
        train_gen = custom_data_generator(
            train_paths_fold,
            train_labels_fold,
            batch_size=batch_size,
            model_name=model_name,
            use_graphic_preprocessing=use_graphic_preprocessing,
            augment=True,
        )
        val_gen = custom_data_generator(
            val_paths_fold,
            val_labels_fold,
            batch_size=batch_size,
            model_name=model_name,
            use_graphic_preprocessing=use_graphic_preprocessing,
            augment=False,
            shuffle=False
        )

        steps_per_epoch = len(train_paths_fold) // batch_size
        validation_steps = len(val_paths_fold) // batch_size

        model_save_path = save_path.replace(".h5", f"_fold_{fold}.h5")
        model, loaded = load_or_build_cnn_model(
            model_name=model_name,
            save_path=model_save_path,
            fine_tune=fine_tune,
            fine_tune_at=fine_tune_at
        )
        if not loaded and fine_tune:
            print(f"[CNN] Treinando (Fold {fold})...")
            model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=validation_steps
            )
            print(f"[CNN] Salvando modelo treinado em {model_save_path}")
            model.save(model_save_path)

        y_val_true = []
        y_val_pred = []
        for _ in range(validation_steps):
            Xb, Yb = next(val_gen)
            preds_b = model.predict(Xb)
            preds_b = np.argmax(preds_b, axis=1)
            true_b = np.argmax(Yb, axis=1)
            y_val_true.extend(true_b)
            y_val_pred.extend(preds_b)

        print(f"[CNN Fold {fold}] Classification Report:")
        print(classification_report(y_val_true, y_val_pred, digits=4))
        print(confusion_matrix(y_val_true, y_val_pred))

        all_preds.extend(y_val_pred)
        all_trues.extend(y_val_true)

        clear_session()
        fold += 1

    print("\n[CNN K-Fold] Relatório Final:")
    print(classification_report(all_trues, all_preds, digits=4))
    print(confusion_matrix(all_trues, all_preds))

###############################################################################
# Extrair features
###############################################################################
def extract_features_with_cnn(
        paths,
        model,
        model_name="VGG19",
        use_graphic_preprocessing=False,
        image_cache=None,
        features_save_path="features.npy",
        batch_size=32
):
    """
    Extrai características de imagens usando um modelo CNN pré-treinado.

    Args:
        paths
        model (keras.Model): Modelo configurado e salvo.
        model_name (str): Nome do modelo ('VGG19', 'Inception', 'ResNet', 'Xception').
        use_graphic_preprocessing (bool): Se True, aplica pré-processamento gráfico.
        image_cache (dict): Dicionário de imagens (cache)
        features_save_path (str): Onde salvar as features extraídas.
        batch_size (int): Batch size.
    Returns:
        features (numpy.ndarray): Vetores de características extraídos.
    """
    if os.path.exists(features_save_path):
        print(f"Carregando features salvas de {features_save_path}")
        return np.load(features_save_path)

    feature_extractor = Model(inputs=model.input, outputs=model.get_layer('dense_features').output)

    print("[Features] Pré-processando imagens...")
    all_images = load_and_preprocess_images(
        paths,
        model_name,
        use_graphic_preprocessing=use_graphic_preprocessing,
        cache=image_cache
    )

    # Extrair características
    print("[Features] Extraindo embeddings...")
    features = feature_extractor.predict(all_images, batch_size=batch_size)

    np.save(features_save_path, features)
    print(f"[Features] Salvas em {features_save_path}")

    return features


###############################################################################
# Classificadores Clássicos
###############################################################################
def train_classical_classifier(
        features,
        labels,
        classifier_type="RandomForest",
        save_path=""
):
    if os.path.exists(save_path):
        from joblib import load
        print(f"Carregando modelo clássico existente: {save_path}")
        return load(save_path)
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
        ], voting='soft')
    else:
        raise ValueError(
            f"Modelo {classifier_type} não suportado. Escolha entre 'RandomForest', 'SVM', 'KNN' ou 'KNN+SVM'.")


    model.fit(features, labels)

    from joblib import dump
    dump(model, save_path)
    print(f"[Classificador] Modelo salvo em: {save_path}")

    return model


def run_kfold_classical(
        all_features,
        all_labels,
        k_folds=5,
        classifier_type="KNN",
        use_data_preprocessing=False,
        pca_n_components=100,
        save_path = ""
):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold = 1
    all_preds = []
    all_trues = []

    for train_idx, val_idx in kf.split(all_features):
        print(f"\n[Classic] Fold {fold}/{k_folds}")
        X_train, X_val = all_features[train_idx], all_features[val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]

        # Pré-process (Scaler + PCA), se desejado
        if use_data_preprocessing:
            pre_save_path = save_path.replace("{model_type}.joblib", "")
            X_train, X_val = pre_process_data(
                train_features=X_train,
                val_features=X_val,
                save_path=pre_save_path,
                n_components=pca_n_components
            )

        # Treina classificador
        clf = train_classical_classifier(X_train, y_train, classifier_type, save_path=save_path)
        preds = clf.predict(X_val)

        print(classification_report(y_val, preds, digits=4))
        print(confusion_matrix(y_val, preds))

        all_preds.extend(preds)
        all_trues.extend(y_val)

        fold += 1

    # Resultado final
    print("\n[Classic] Relatório Final:")
    print(classification_report(all_trues, all_preds, digits=4))
    print(confusion_matrix(all_trues, all_preds))

###############################################################################
# Função Principal
###############################################################################
def run_experiment(
        model_name="VGG19",
        use_graphic_preprocessing=False,
        use_data_preprocessing=False,
        use_cnn_classifier=True,
        classical_classifier="RandomForest",
        fine_tune=False,
        fine_tune_at=0,
        k_folds=5,
        pca_n_components=100,
        epochs=10,
        batch_size=64,
        num_classes=7
):
    """
    Função principal dinâmica:
      - Se `use_cnn_classifier=True`, roda a CNN como classificador.
        *opcionalmente* com K-Fold (end-to-end).
      - Se `use_classical_classifier=True`, usa a CNN como extratora de
        features e roda K-Fold nos modelos clássicos.
      - Você pode ajustar hiperparâmetros via estes argumentos.
    """
    # 1) Carrega paths e labels
    train_paths, train_labels = load_paths_labels("../res/train_files.txt")
    val_paths, val_labels = load_paths_labels("../res/val_files.txt")

    # Concatenar se quiser um único dataset
    all_paths = np.concatenate([train_paths, val_paths])
    all_labels = np.concatenate([train_labels, val_labels])

    image_cache = {}
    model_type = "classifier" if use_cnn_classifier else "extractor"
    save_path = f"../models/CNN/{model_name}/{model_name.lower()}_graphic_{use_graphic_preprocessing}_fine_tune_{fine_tune}{f'_fine_tune_at_{fine_tune_at}' if fine_tune_at != 0 else ''}_{model_type}.h5"

    # 2) CENÁRIO A: CNN como Classificador
    #    Se `use_cnn_classifier`

    if use_cnn_classifier:
        # CNN como classificador
        print("\n===== [Modo] CNN como Classificador End-to-End =====")
        print(f"\nCenário - Modelo: {model_name}, Fine-Tune: {fine_tune}, {f'Camada: {fine_tune_at}, ' if fine_tune_at != 0 else ''}Pré-Processamento Gráfico: {use_graphic_preprocessing}")


        if k_folds > 1:
            # (A1) Rodar K-Fold na CNN
            run_kfold_cnn(
                all_paths=all_paths,
                all_labels=all_labels,
                model_name=model_name,
                fine_tune=fine_tune,
                fine_tune_at=fine_tune_at,
                k_folds=k_folds,
                epochs=epochs,
                batch_size=batch_size,
                use_graphic_preprocessing=use_graphic_preprocessing,
                save_path=save_path
            )
        else:
            # (A2) Apenas um split fixo
            print("[CNN Single-Split] Treinando rede apenas uma vez.")
            # Exemplo: 80% train, 20% val
            split_point = int(0.20 * len(all_paths))
            cnn_train_paths = all_paths[:split_point]
            cnn_val_paths = all_paths[split_point:]
            cnn_train_labels = all_labels[:split_point]
            cnn_val_labels = all_labels[split_point:]

            train_gen = custom_data_generator(
                paths=cnn_train_paths,
                labels=cnn_train_labels,
                batch_size=batch_size,
                model_name=model_name,
                use_graphic_preprocessing=use_graphic_preprocessing,
                use_hair_removal=True,  # se quiser
                shuffle=True,
                augment=True  # se quiser data augmentation
            )
            val_gen = custom_data_generator(
                paths=cnn_val_paths,
                labels=cnn_val_labels,
                batch_size=batch_size,
                model_name=model_name,
                use_graphic_preprocessing=use_graphic_preprocessing,
                use_hair_removal=True,
                shuffle=False,
                augment=False
            )
            steps_per_epoch = len(cnn_train_paths) // batch_size
            validation_steps = len(cnn_val_paths) // batch_size

            # Treina ou carrega
            model, loaded = load_or_build_cnn_model(
                model_name=model_name,
                save_path=save_path,
                fine_tune=fine_tune,
                fine_tune_at=fine_tune_at,
                num_classes=num_classes
            )
            if not loaded and fine_tune:
                print("[CNN Single-Split] Treinando...")
                model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=val_gen,
                    validation_steps=validation_steps
                )
                print(f"[CNN Single-Split] Salvando modelo treinado em {save_path}")
                model.save(save_path)

            print("[CNN Single-Split] Avaliando no conjunto de validação (gerador)...")
            val_gen_eval = custom_data_generator(
                paths=cnn_val_paths,
                labels=cnn_val_labels,
                batch_size=batch_size,
                model_name=model_name,
                use_graphic_preprocessing=use_graphic_preprocessing,
                shuffle=False,
                augment=False
            )
            steps_val = math.ceil(len(cnn_val_paths) / batch_size)
            y_true_all = []
            y_pred_all = []
            for _ in range(steps_val):
                Xb, Yb = next(val_gen_eval)
                preds_b = model.predict(Xb)
                preds_b = np.argmax(preds_b, axis=1)
                true_b = np.argmax(Yb, axis=1)
                y_true_all.extend(true_b)
                y_pred_all.extend(preds_b)

            print("[CNN Single-Split] Classification Report:")
            print(classification_report(y_true_all, y_pred_all, digits=4))
            print(confusion_matrix(y_true_all, y_pred_all))
    else:
        # CNN como extratora de características
        print("\n===== [Modo] CNN como Extratora + Classificador Clássico =====")
        print(f"\nCenário - Classificador: {classical_classifier}, CNN: {model_name}, Fine-Tune: {fine_tune}, {f'Camada: {fine_tune_at}, ' if fine_tune_at != 0 else ''}Pré-Processamento Gráfico: {use_graphic_preprocessing}, Pré-Processamento de Dados: {use_data_preprocessing}")

        clf_save_path = f"../models/CLASSIC/{classical_classifier}/{classical_classifier.lower()}_{model_name.lower()}_fine_tune_{fine_tune}{f'_fine_tune_at_{fine_tune_at}' if fine_tune_at != 0 else ''}_graphic_{use_graphic_preprocessing}/{model_type}.joblib"
        features_save_path = f"../models/CNN/{model_name}/{model_name.lower()}_graphic_{use_graphic_preprocessing}_fine_tune_{fine_tune}{f'_fine_tune_at_{fine_tune_at}' if fine_tune_at != 0 else ''}_extractor_features.npy"

        split_point = int(0.20 * len(all_paths))
        cnn_train_paths = all_paths[:split_point]
        cnn_val_paths = all_paths[split_point:]
        cnn_train_labels = all_labels[:split_point]
        cnn_val_labels = all_labels[split_point:]

        x_cnn_train, y_cnn_train = load_and_preprocess_images(
            cnn_train_paths, model_name, labels=cnn_train_labels,
            use_graphic_preprocessing=use_graphic_preprocessing,
            cache=image_cache
        )
        x_cnn_val, y_cnn_val = load_and_preprocess_images(
            cnn_val_paths, model_name, labels=cnn_val_labels,
            use_graphic_preprocessing=use_graphic_preprocessing,
            cache=image_cache
        )

        model, loaded = load_or_build_cnn_model(
            model_name=model_name,
            save_path=save_path,
            fine_tune=fine_tune,
            fine_tune_at=fine_tune_at,
            num_classes=num_classes
        )

        if fine_tune and not loaded:
            print("[CNN Extractor] Treinando CNN p/ extrair features...")
            model.fit(
                x_cnn_train, y_cnn_train,
                validation_data=(x_cnn_val, y_cnn_val),
                epochs=epochs,
                batch_size=batch_size
            )
            print(f"[CNN Extractor] Salvando modelo treinado em {save_path}")
            model.save(save_path)

        all_features = extract_features_with_cnn(
            model=model,
            paths=all_paths,
            model_name=model_name,
            use_graphic_preprocessing=use_graphic_preprocessing,
            features_save_path=features_save_path
        )

        run_kfold_classical(
            all_features=all_features,
            all_labels=all_labels,
            k_folds=k_folds,
            classifier_type=classical_classifier,
            use_data_preprocessing=use_data_preprocessing,
            pca_n_components=pca_n_components,
            save_path=clf_save_path
        )

def main():
    with tensorflow.device('/GPU:0'):
        run_experiment(
            model_name="VGG19",
            use_cnn_classifier=True,
            use_graphic_preprocessing=True,
            use_data_preprocessing=True,
            classical_classifier="RandomForest",
            fine_tune=True,
            fine_tune_at=0,
            k_folds=5,
            pca_n_components=100,
            epochs=skincancer.src.config.num_epochs,
            batch_size=skincancer.src.config.batch_size,
            num_classes=skincancer.src.config.num_classes
        )

if __name__ == '__main__':
    main()











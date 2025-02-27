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
from tensorflow.keras.callbacks import EarlyStopping

import skincancer.src.config
from utils import (
    load_paths_labels,
    load_and_preprocess_images,
    custom_data_generator
)

early_stopping = EarlyStopping(
    monitor='val_loss',  # Métrica a ser monitorada (pode ser 'val_accuracy', se preferir)
    patience=10,  # Número de epochs a esperar sem melhora antes de interromper o treinamento
    restore_best_weights=True  # Restaura os pesos do modelo que obtiveram a melhor performance na validação
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
def load_or_build_cnn_model(fine_tune_at=0, save_path=""):
    """
    Cria ou carrega um modelo CNN e o salva em 'save_path'.
    Se já existir 'save_path', o modelo é carregado.

    Args:
        fine_tune_at (int): Camada para descongelar.
        save_path (str): Caminho em que deve ser salvo o modelo.

    Returns:
        model (keras.Model): Modelo configurado e salvo.
        loaded (bool): Define se o modelo foi criado ou se foi carregado.
    """
    if os.path.exists(save_path):
        print(f"Carregando modelo existente: {save_path}")
        return load_model(save_path), True

    if skincancer.src.config.cnn_model == "VGG19":
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=skincancer.src.config.img_size)
        fine_tune_at = fine_tune_at if fine_tune_at != 0 else 15
    elif skincancer.src.config.cnn_model == "Inception":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=skincancer.src.config.img_size)
        fine_tune_at = fine_tune_at if fine_tune_at != 0 else 40
    elif skincancer.src.config.cnn_model == "ResNet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=skincancer.src.config.img_size)
        fine_tune_at = fine_tune_at if fine_tune_at != 0 else 25
    elif skincancer.src.config.cnn_model == "Xception":
        base_model = Xception(weights='imagenet', include_top=False, input_shape=skincancer.src.config.img_size)
        fine_tune_at = fine_tune_at if fine_tune_at != 0 else 30
    else:
        raise ValueError(
            f"Modelo {skincancer.src.config.cnn_model} não suportado. Escolha entre 'VGG19', 'Inception', 'ResNet' ou 'Xception'.")

    if skincancer.src.config.use_fine_tuning:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        base_model.trainable = False

    x = layers.GlobalAveragePooling2D(name='gap')(base_model.output)
    x = layers.Dense(256, activation='relu', name='dense_features')(x)
    x = layers.Dropout(0.5, name='dropout_features')(x)
    predictions = layers.Dense(skincancer.src.config.num_classes, activation='softmax', name='final_predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    return model, False

###############################################################################
# K-Fold CNN
###############################################################################
def run_kfold_cnn(
        all_paths,
        all_labels,
        save_path=""
):
    """
    Exemplo de K-Fold end-to-end na CNN.
    Em cada fold, treina uma CNN do zero (ou com fine-tune).
    CUIDADO: Alto custo computacional se k_folds > 1, pois treinará
             a CNN repetidas vezes.
    """
    kf = KFold(n_splits=skincancer.src.config.num_kfolds, shuffle=True, random_state=42)
    fold = 1

    all_preds = []
    all_trues = []

    for train_idx, val_idx in kf.split(all_paths):
        print(f"\n[CNN K-Fold] Fold {fold}/{skincancer.src.config.num_kfolds}")
        train_paths_fold  = all_paths[train_idx]
        val_paths_fold    = all_paths[val_idx]
        train_labels_fold = all_labels[train_idx]
        val_labels_fold   = all_labels[val_idx]

        # Cria geradores
        train_gen = custom_data_generator(
            train_paths_fold,
            train_labels_fold,
            batch_size=skincancer.src.config.batch_size,
            model_name=skincancer.src.config.cnn_model,
            use_graphic_preprocessing=skincancer.src.config.use_graphic_preprocessing,
            augment=skincancer.src.config.use_data_augmentation,
        )
        val_gen = custom_data_generator(
            val_paths_fold,
            val_labels_fold,
            batch_size=skincancer.src.config.batch_size,
            model_name=skincancer.src.config.cnn_model,
            use_graphic_preprocessing=skincancer.src.config.use_graphic_preprocessing,
            augment=False,
            shuffle=False
        )

        steps_per_epoch = math.ceil(len(train_paths_fold) / skincancer.src.config.batch_size)
        validation_steps = math.ceil(len(val_paths_fold) / skincancer.src.config.batch_size)

        model_save_path = save_path.replace(".h5", f"_fold_{fold}.h5")
        model, loaded = load_or_build_cnn_model(fine_tune_at=skincancer.src.config.use_fine_tuning_at_layer, save_path=model_save_path)
        if not loaded:
            print(f"[CNN] Treinando (Fold {fold})...")

            from sklearn.utils import class_weight
            import numpy as np
            class_weights_array = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels_fold),
                y=train_labels_fold
            )
            class_weights = dict(enumerate(class_weights_array))

            model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=skincancer.src.config.num_epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=[early_stopping],
                class_weight=class_weights if skincancer.src.config.use_class_weight else None
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
        features_save_path="features.npy"
):
    """
    Extrai características de imagens usando um modelo CNN pré-treinado.

    Args:
        paths
        model (keras.Model): Modelo configurado e salvo.
        features_save_path (str): Onde salvar as features extraídas.
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
        skincancer.src.config.cnn_model,
        use_graphic_preprocessing=skincancer.src.config.use_graphic_preprocessing
    )

    # Extrair características
    print("[Features] Extraindo embeddings...")
    features = feature_extractor.predict(all_images, batch_size=skincancer.src.config.batch_size)

    if features_save_path.endswith(".npy"):
        np.save(features_save_path, features)
        print(f"[Features] Salvas em {features_save_path}")

    return features


###############################################################################
# Classificadores Clássicos
###############################################################################
def train_classical_classifier(
        features,
        labels,
        save_path=""
):
    if os.path.exists(save_path):
        from joblib import load
        print(f"Carregando modelo clássico existente: {save_path}")
        return load(save_path)
    if skincancer.src.config.classical_classifier_model == "RandomForest":
        model = RandomForestClassifier(random_state=42, class_weight="balanced" if skincancer.src.config.use_class_weight else None)
    elif skincancer.src.config.classical_classifier_model == "SVM":
        model = SVC(probability=True, random_state=42, class_weight="balanced" if skincancer.src.config.use_class_weight else None)
    elif skincancer.src.config.classical_classifier_model == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif skincancer.src.config.classical_classifier_model == "KNN+SVM":
        knn = KNeighborsClassifier(n_neighbors=5)
        svm = SVC(probability=True, random_state=42, class_weight="balanced" if skincancer.src.config.use_class_weight else None)
        model = VotingClassifier(estimators=[
            ('knn', knn),
            ('svm', svm)
        ], voting='soft')
    else:
        raise ValueError(
            f"Modelo {skincancer.src.config.classical_classifier_model} não suportado. Escolha entre 'RandomForest', 'SVM', 'KNN' ou 'KNN+SVM'.")


    model.fit(features, labels)

    from joblib import dump
    dump(model, save_path)
    print(f"[Classificador] Modelo salvo em: {save_path}")

    return model


def run_kfold_classical(
        all_features,
        all_labels,
        save_path = ""
):
    kf = KFold(n_splits=skincancer.src.config.num_kfolds, shuffle=True, random_state=42)

    fold = 1
    all_preds = []
    all_trues = []

    for train_idx, val_idx in kf.split(all_features):
        print(f"\n[Classic] Fold {fold}/{skincancer.src.config.num_kfolds}")
        X_train, X_val = all_features[train_idx], all_features[val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]

        # Pré-process (Scaler + PCA), se desejado
        if skincancer.src.config.use_data_preprocessing:
            pre_save_path = save_path.replace("{model_type}.joblib", "")
            X_train, X_val = pre_process_data(
                train_features=X_train,
                val_features=X_val,
                save_path=pre_save_path,
                n_components=skincancer.src.config.num_pca_components
            )

        # Treina classificador
        clf = train_classical_classifier(X_train, y_train, save_path=save_path)
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
def run_experiment():
    # 1) Carrega paths e labels
    train_paths, train_labels = load_paths_labels("../res/train_files.txt")
    val_paths, val_labels = load_paths_labels("../res/val_files.txt")
    test_paths, test_labels = load_paths_labels("../res/test_files.txt")

    model_type = "classifier" if skincancer.src.config.use_cnn_as_classifier else "extractor"

    cnn_save_path      = f"../models/{model_type}/use_graphic_preprocessing_{skincancer.src.config.use_graphic_preprocessing}/use_data_augmentation_{skincancer.src.config.use_data_augmentation}/use_fine_tuning_{skincancer.src.config.use_fine_tuning}/use_class_weights_{skincancer.src.config.use_class_weight}/{skincancer.src.config.cnn_model.lower()}/{model_type}.h5"
    clf_save_path      = f"../models/{model_type}/use_graphic_preprocessing_{skincancer.src.config.use_graphic_preprocessing}/use_data_augmentation_{skincancer.src.config.use_data_augmentation}/use_fine_tuning_{skincancer.src.config.use_fine_tuning}/use_class_weights_{skincancer.src.config.use_class_weight}/{skincancer.src.config.classical_classifier_model.lower()}_{skincancer.src.config.cnn_model.lower()}/{model_type}.joblib"
    features_save_path = f"../models/{model_type}/use_graphic_preprocessing_{skincancer.src.config.use_graphic_preprocessing}/use_data_augmentation_{skincancer.src.config.use_data_augmentation}/use_fine_tuning_{skincancer.src.config.use_fine_tuning}/use_class_weights_{skincancer.src.config.use_class_weight}/{skincancer.src.config.cnn_model.lower()}/extractor_features.npy"

    # 2) CENÁRIO A: CNN como Classificador

    if skincancer.src.config.use_cnn_as_classifier:
        # CNN como classificador
        print("\n===== [Modo] CNN como Classificador End-to-End =====")

        if skincancer.src.config.num_kfolds > 1:
            # (A1) Rodar K-Fold na CNN
            run_kfold_cnn(
                all_paths=train_paths,
                all_labels=train_labels,
                save_path=cnn_save_path
            )
        else:
            # (A2) Apenas um split fixo
            print("[CNN Single-Split] Treinando rede apenas uma vez.")

            train_gen = custom_data_generator(
                paths=train_paths,
                labels=train_labels,
                batch_size=skincancer.src.config.batch_size,
                model_name=skincancer.src.config.cnn_model,
                use_graphic_preprocessing=skincancer.src.config.use_graphic_preprocessing,
                use_hair_removal=skincancer.src.config.use_graphic_preprocessing,
                shuffle=skincancer.src.config.use_data_augmentation,
                augment=skincancer.src.config.use_data_augmentation
            )
            val_gen = custom_data_generator(
                paths=val_paths,
                labels=val_labels,
                batch_size=skincancer.src.config.batch_size,
                model_name=skincancer.src.config.cnn_model,
                use_graphic_preprocessing=skincancer.src.config.use_graphic_preprocessing,
                use_hair_removal=skincancer.src.config.use_graphic_preprocessing,
                shuffle=False,
                augment=False
            )
            steps_per_epoch = math.ceil(len(train_paths) / skincancer.src.config.batch_size)
            validation_steps = math.ceil(len(val_paths) / skincancer.src.config.batch_size)

            # Treina ou carrega
            model, loaded = load_or_build_cnn_model(fine_tune_at=skincancer.src.config.use_fine_tuning_at_layer, save_path=cnn_save_path)
            if not loaded:
                print("[CNN Single-Split] Treinando...")
                from sklearn.utils import class_weight
                class_weights_array = class_weight.compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(train_labels),
                    y=train_labels
                )
                class_weights = dict(enumerate(class_weights_array))
                model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=skincancer.src.config.num_epochs,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    callbacks=[early_stopping],
                    class_weight=class_weights if skincancer.src.config.use_class_weight else None
                )
                print(f"[CNN Single-Split] Salvando modelo treinado em {cnn_save_path}")
                model.save(cnn_save_path)

            # Avaliação final no conjunto de teste (virgem)
            test_gen = custom_data_generator(
                paths=test_paths,
                labels=test_labels,
                batch_size=skincancer.src.config.batch_size,
                model_name=skincancer.src.config.cnn_model,
                use_graphic_preprocessing=skincancer.src.config.use_graphic_preprocessing,
                shuffle=False,
                augment=False
            )
            test_steps = math.ceil(len(test_paths) / skincancer.src.config.batch_size)
            y_true_test = []
            y_pred_test = []
            for _ in range(test_steps):
                Xb, Yb = next(test_gen)
                preds_b = model.predict(Xb)
                preds_b = np.argmax(preds_b, axis=1)
                true_b = np.argmax(Yb, axis=1)
                y_true_test.extend(true_b)
                y_pred_test.extend(preds_b)

            print("[CNN Single-Split] Relatório Final no conjunto de TESTE:")
            print(classification_report(y_true_test, y_pred_test, digits=4))
            print(confusion_matrix(y_true_test, y_pred_test))
    else:
        # CNN como extratora de características
        print("\n===== [Modo] CNN como Extratora + Classificador Clássico =====")

        x_train, y_train = load_and_preprocess_images(
            train_paths, skincancer.src.config.cnn_model, labels=train_labels,
            use_graphic_preprocessing=skincancer.src.config.use_graphic_preprocessing,
        )
        x_val, y_val = load_and_preprocess_images(
            train_paths, skincancer.src.config.cnn_model, labels=train_labels,
            use_graphic_preprocessing=skincancer.src.config.use_graphic_preprocessing,
        )
        x_test, y_test = load_and_preprocess_images(
            test_paths, skincancer.src.config.cnn_model, labels=test_labels,
            use_graphic_preprocessing=skincancer.src.config.use_graphic_preprocessing
        )

        model, loaded = load_or_build_cnn_model(fine_tune_at=skincancer.src.config.use_fine_tuning_at_layer, save_path=cnn_save_path)

        if not loaded:
            print("[CNN Extractor] Treinando CNN p/ extrair features...")

            model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=skincancer.src.config.num_epochs,
                batch_size=skincancer.src.config.batch_size,
                callbacks=[early_stopping]
            )
            print(f"[CNN Extractor] Salvando modelo treinado em {cnn_save_path}")
            model.save(cnn_save_path)

        all_features = extract_features_with_cnn(
            paths=np.concatenate([train_paths, val_paths]),
            model=model,
            features_save_path=features_save_path
        )

        run_kfold_classical(
            all_features=all_features,
            all_labels=np.concatenate([train_labels, val_labels]),
            save_path=clf_save_path
        )

        # Avaliação final no conjunto de teste para o classificador clássico
        test_features = extract_features_with_cnn(
            paths=test_paths,
            model=model,
            features_save_path=""
        )
        from joblib import load
        clf = load(clf_save_path)
        test_preds = clf.predict(test_features)
        print("\n[Classic] Relatório Final no conjunto de TESTE:")
        print(classification_report(test_labels, test_preds, digits=4))
        print(confusion_matrix(test_labels, test_preds))

def main():
    with tensorflow.device('/GPU:0'):
        run_experiment()

if __name__ == '__main__':
    main()











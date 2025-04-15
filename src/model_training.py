import gc
import math
import os

import numpy as np
import tensorflow
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.metrics import Recall
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG19, InceptionV3, ResNet50, Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import GlobalAveragePooling2D

import skincancer.src.config
from utils import (
    load_paths_labels,
    load_and_preprocess_images,
    data_generator
)


def get_callbacks(save_path):
    return [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint(filepath=save_path, save_best_only=True)
    ]


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
        fine_tune_at = 15
    elif skincancer.src.config.cnn_model == "Inception":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=skincancer.src.config.img_size)
        fine_tune_at = 280
    elif skincancer.src.config.cnn_model == "ResNet":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=skincancer.src.config.img_size)
        fine_tune_at = 140
    elif skincancer.src.config.cnn_model == "Xception":
        base_model = Xception(weights='imagenet', include_top=False, input_shape=skincancer.src.config.img_size)
        fine_tune_at = 100
    else:
        raise ValueError(
            f"Modelo {skincancer.src.config.cnn_model} não suportado. Escolha entre 'VGG19', 'Inception', 'ResNet' ou 'Xception'.")


    if skincancer.src.config.use_fine_tuning:
        print(f"[FINE-TUNE] treinando rede a partir da camada: {fine_tune_at}")
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            layer.trainable = True
    else:
        base_model.trainable = False

    if skincancer.src.config.use_cnn_as_classifier:
        x = layers.GlobalAveragePooling2D(name='gap')(base_model.output)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(skincancer.src.config.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy', Recall()])
    else:
        model = base_model

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
        train_gen = data_generator(
            train_paths_fold,
            train_labels_fold,
            batch_size=skincancer.src.config.batch_size,
            model_name=skincancer.src.config.cnn_model,
            augment=skincancer.src.config.use_data_preprocessing
        )
        val_gen = data_generator(
            val_paths_fold,
            val_labels_fold,
            batch_size=skincancer.src.config.batch_size,
            model_name=skincancer.src.config.cnn_model,
            augment=False,
            shuffle=False
        )

        steps_per_epoch = math.ceil(len(train_paths_fold) / skincancer.src.config.batch_size)
        validation_steps = math.ceil(len(val_paths_fold) / skincancer.src.config.batch_size)

        model_save_path = save_path.replace(".h5", f"_fold_{fold}.h5")
        model, loaded = load_or_build_cnn_model(fine_tune_at=skincancer.src.config.use_fine_tuning_at_layer, save_path=model_save_path)
        if not loaded:
            print(f"[CNN] Treinando (Fold {fold})...")
            model.fit(
                train_gen,
                steps_per_epoch=steps_per_epoch,
                epochs=skincancer.src.config.num_epochs,
                validation_data=val_gen,
                validation_steps=validation_steps,
                callbacks=get_callbacks(model_save_path)
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
        features_save_path="features.npy",
        augment=False
):
    if os.path.exists(features_save_path):
        print(f"Carregando features salvas de {features_save_path}")
        return np.load(features_save_path)

    layer_name = {
        "VGG19": "block5_pool",
        "Inception": "mixed10",
        "ResNet": "avg_pool",
        "Xception": "avg_pool"
    }[skincancer.src.config.cnn_model]

    base_output = model.get_layer(layer_name).output

    gap_output = GlobalAveragePooling2D()(base_output)

    feature_extractor = Model(inputs=model.input, outputs=gap_output)

    print("[Features] Pré-processando imagens...")
    all_images = load_and_preprocess_images(
        paths,
        skincancer.src.config.cnn_model,
        segmentation=False,
        augment=augment
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
def get_classifier():
    import skincancer.src.config as config
    if config.classical_classifier_model == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
    elif config.classical_classifier_model == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=42, eval_metric='mlogloss')
    elif config.classical_classifier_model == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(random_state=42)
    elif config.classical_classifier_model == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(random_state=42)
    else:
        raise ValueError("Modelo de classificador clássico não suportado. Escolha entre 'RandomForest', 'XGBoost', 'AdaBoost' ou 'ExtraTrees'.")
    return clf

def run_kfold_classical(all_features, all_labels, save_path=""):
    import skincancer.src.config as config
    from sklearn.model_selection import KFold
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    import os

    kf = KFold(n_splits=config.num_kfolds, shuffle=True, random_state=42)
    fold = 1
    fold_scores = []

    for train_idx, val_idx in kf.split(all_features):
        print(f"\n[Classic] Fold {fold}/{config.num_kfolds}")
        X_train, X_val = all_features[train_idx], all_features[val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]

        X_train_res, y_train_res = X_train, y_train

        clf = get_classifier()
        clf.fit(X_train_res, y_train_res)
        preds = clf.predict(X_val)
        print(classification_report(y_val, preds, digits=4))
        print(confusion_matrix(y_val, preds))
        score = clf.score(X_val, y_val)
        fold_scores.append(score)
        fold += 1

    print("\n[Classic] Relatório Final de Cross-Validation:")
    print(f"Média de acurácia: {np.mean(fold_scores):.4f}")

    clear_session()
    gc.collect()

    print("\nRe-treinando classificador com o conjunto completo de treinamento...")
    clf_final = get_classifier()
    clf_final.fit(all_features, all_labels)
    print(f"[Classic] Salvando classificador final em: {save_path}")
    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    joblib.dump(clf_final, save_path)


def evaluate_classical_classifier(test_features, test_labels, classifier_path):
    from joblib import load
    from sklearn.metrics import classification_report, confusion_matrix

    # Carrega o classificador salvo
    classifier = load(classifier_path)

    # Realiza as predições no conjunto de teste
    preds = classifier.predict(test_features)

    # Imprime o relatório de classificação e a matriz de confusão
    print("\n[Classic Classifier] Relatório Final no conjunto de TESTE:")
    print(classification_report(test_labels, preds, digits=4))
    print(confusion_matrix(test_labels, preds))

###############################################################################
# Função Principal
###############################################################################
def run_experiment():
    # 1) Carrega paths e labels
    train_paths, train_labels = load_paths_labels("../res/train_files.txt")
    val_paths, val_labels = load_paths_labels("../res/val_files.txt")
    test_paths, test_labels = load_paths_labels("../res/test_files.txt")

    model_type = "classifier" if skincancer.src.config.use_cnn_as_classifier else "extractor"

    cnn_save_path      = f"../models/{model_type}/use_graphic_preprocessing_{skincancer.src.config.use_graphic_preprocessing}/use_data_preprocessing_{skincancer.src.config.use_data_preprocessing}/use_fine_tuning_{skincancer.src.config.use_fine_tuning}/{skincancer.src.config.cnn_model.lower()}/{model_type}.h5"
    clf_save_path      = f"../models/{model_type}/use_graphic_preprocessing_{skincancer.src.config.use_graphic_preprocessing}/use_data_preprocessing_{skincancer.src.config.use_data_preprocessing}/use_fine_tuning_{skincancer.src.config.use_fine_tuning}/{skincancer.src.config.cnn_model.lower()}/{skincancer.src.config.classical_classifier_model.lower()}/classifier.joblib"
    features_save_path = f"../models/{model_type}/use_graphic_preprocessing_{skincancer.src.config.use_graphic_preprocessing}/use_data_preprocessing_{skincancer.src.config.use_data_preprocessing}/use_fine_tuning_{skincancer.src.config.use_fine_tuning}/{skincancer.src.config.cnn_model.lower()}/extracted_features.npy"

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

            train_gen = data_generator(
                paths=train_paths,
                labels=train_labels,
                batch_size=skincancer.src.config.batch_size,
                model_name=skincancer.src.config.cnn_model,
                augment=skincancer.src.config.use_data_preprocessing
            )
            val_gen = data_generator(
                paths=val_paths,
                labels=val_labels,
                batch_size=skincancer.src.config.batch_size,
                model_name=skincancer.src.config.cnn_model,
                shuffle=False,
                augment=False
            )
            steps_per_epoch = math.ceil(len(train_paths) / skincancer.src.config.batch_size)
            validation_steps = math.ceil(len(val_paths) / skincancer.src.config.batch_size)

            # Treina ou carrega
            model, loaded = load_or_build_cnn_model(fine_tune_at=skincancer.src.config.use_fine_tuning_at_layer, save_path=cnn_save_path)
            if not loaded:
                print("[CNN Single-Split] Treinando...")
                model.fit(
                    train_gen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=skincancer.src.config.num_epochs,
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    callbacks=get_callbacks(cnn_save_path)
                )
                print(f"[CNN Single-Split] Salvando modelo treinado em {cnn_save_path}")
                model.save(cnn_save_path)

            # Avaliação final no conjunto de teste (virgem)
            test_gen = data_generator(
                paths=test_paths,
                labels=test_labels,
                batch_size=skincancer.src.config.batch_size,
                model_name=skincancer.src.config.cnn_model,
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
        model, loaded = load_or_build_cnn_model(fine_tune_at=skincancer.src.config.use_fine_tuning_at_layer, save_path=cnn_save_path)
        if not loaded and skincancer.src.config.use_fine_tuning:
            print("[CNN Extractor] Treinando CNN p/ extrair features...")

            x_train, y_train = load_and_preprocess_images(
                train_paths,
                skincancer.src.config.cnn_model,
                labels=train_labels,
                segmentation=False,
                augment=skincancer.src.config.use_data_preprocessing,
                visualize=skincancer.src.config.visualize
            )
            x_val, y_val = load_and_preprocess_images(
                val_paths,
                skincancer.src.config.cnn_model,
                labels=val_labels,
                segmentation=False,
                augment=False
            )

            x = GlobalAveragePooling2D()(model.output)
            softmax_output = layers.Dense(skincancer.src.config.num_classes, activation="softmax")(x)
            ft_model = Model(inputs=model.input, outputs=softmax_output)

            ft_model.compile(
                optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            ft_model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=skincancer.src.config.num_epochs,
                batch_size=skincancer.src.config.batch_size,
                callbacks=get_callbacks(cnn_save_path)
            )

            embedding_output = ft_model.layers[-2].output
            feature_model = Model(inputs=ft_model.input, outputs=embedding_output)

            print(f"[CNN Extractor] Salvando modelo treinado em {cnn_save_path}")
            feature_model.save(cnn_save_path)

            clear_session()
            gc.collect()

            model = feature_model

        features_train = extract_features_with_cnn(
            paths=train_paths,
            model=model,
            features_save_path=features_save_path.replace("extracted_features.npy", "extracted_features_train.npy"),
            augment=True,
        )
        features_val = extract_features_with_cnn(
            paths=val_paths,
            model=model,
            features_save_path=features_save_path.replace("extracted_features.npy", "extracted_features_val.npy"),
            augment=False,
        )

        all_features = np.concatenate([features_train, features_val])

        run_kfold_classical(
            all_features=all_features,
            all_labels=np.concatenate([train_labels, val_labels]),
            save_path=clf_save_path,
        )

        # Avaliação final no conjunto de teste para o classificador clássico
        test_features = extract_features_with_cnn(
            paths=test_paths,
            model=model,
            features_save_path="",
            augment=False
        )

        evaluate_classical_classifier(test_features, test_labels, clf_save_path)

def main():
    with tensorflow.device('/GPU:0'):
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        run_experiment()

if __name__ == '__main__':
    main()











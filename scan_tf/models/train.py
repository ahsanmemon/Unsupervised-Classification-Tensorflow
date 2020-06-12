import tensorflow as tf
import numpy as np
import scipy
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import scan_tf.models.resnet as resnet
import scan_tf.utils.utils as utils


def pretext_training(backbone_model, X_train, y_train, epochs=100, save_path=None):
    lr = 1e-4
    m = X_train.shape[0]
    batch_size = 128
    pretext_model = resnet.add_classification_layer(backbone_model, 4)
    # optimizer = tf.keras.optimizers.Adam(lr=lr)
    optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=5e-4)
    for e in range(epochs):
        for batch_count in range(m // batch_size):  # batch
            batch_ids = np.random.choice(range(0, m), batch_size)
            images = tf.convert_to_tensor(X_train[batch_ids, ...])
            labels = tf.convert_to_tensor(y_train[batch_ids, ...])
            if True:
                images_rotated = []
                for image in images:
                    images_rotated += [np.rot90(image, i) for i in [0, 1, 2, 3]]
                images_rotated = np.array(images_rotated)
                labels = tf.convert_to_tensor(to_categorical([0, 1, 2, 3]*batch_size))

                # degrees_to_rotate = np.random.choice([0, 1, 2, 3], batch_size)  # randomly choosing the rotation angle
                # images_rotated = np.array([np.rot90(images[i], degrees_to_rotate[i]) for i in range(len(degrees_to_rotate))])
                # images_rotated = tf.convert_to_tensor(images_rotated)
                # labels = tf.convert_to_tensor(to_categorical(degrees_to_rotate))

                with tf.GradientTape() as tape:
                    predictions = pretext_model(images_rotated)
                    loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(labels, predictions, from_logits=False))
                    grads = tape.gradient(loss, pretext_model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, pretext_model.trainable_weights))
            # Disregard all this
            else:
                images_augmented = tf.convert_to_tensor(
                    augmentations.strong_augmentation_ims(X_train[idx:idx + batch_size, ...], is_array=True))
                with tf.GradientTape() as tape:
                    embs_images = backbone_model(images)
                    embs_images_augmented = backbone_model(images_augmented)
                    norm = tf.map_fn(tf.nn.l2_loss, embs_images - embs_images_augmented)
                    loss = tf.reduce_mean(norm)
                    grads = tape.gradient(loss, backbone_model.trainable_weights)
                    optimizer = tf.keras.optimizers.Adam(lr=lr)
                    optimizer.apply_gradients(zip(grads, backbone_model.trainable_weights))

            if np.isnan(loss):
                print("Nan loss found:")
                print(f"epoch {e}, batch {batch_count} loss: {loss}")
                return pretext_model

        if e % 1 == 0:
            # clear_output()
            print(f"{np.min(predictions)},{np.max(predictions)}")
            print(f"{images_rotated.shape}, {labels.shape}")
            print(f"epoch {e}, loss: {loss}")

    if save_path is not None:
        pretext_model.save_weights(save_path)
    return pretext_model


def pretext_training_bis(backbone_model, X_train, y_train, epochs=100, save_path=None):
    lr = 1e-4
    m = X_train.shape[0]
    batch_size = 128
    pretext_model = resnet.add_classification_layer(backbone_model, 4)

    # Prepare data
    images_rotated = []
    for image in X_train:
        images_rotated += [np.rot90(image, i) for i in [0, 1, 2, 3]]
    images_rotated = np.array(images_rotated)
    labels = to_categorical([0, 1, 2, 3] * len(X_train))

    # optimizer = tf.keras.optimizers.Adam(lr=lr)
    optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=5e-4)
    pretext_model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'],
                          )
    pretext_model.fit(images_rotated, labels, epochs=epochs, batch_size=batch_size, shuffle=False)

    if save_path is not None:
        pretext_model.save_weights(save_path)
    return pretext_model


def semantic_clustering_training(backbone_model, X_train, y_train, num_clusters=None, epochs=100, save_path=None):
    n_neighbors = 6
    lr = 1e-4
    m = X_train.shape[0]
    batch_size = 64
    lam = 1

    optimizer = tf.keras.optimizers.Adam(lr=lr)

    if num_clusters is None:
        num_clusters = y_train.shape[1]
    clustering_model = resnet.add_classification_layer(backbone_model, num_clusters)

    # Get clusters
    nn = utils.CLusteringNN(clustering_model, n_neighbors=n_neighbors)
    nn.fit(X_train)
    nn_indexes = nn.get_neighbors_indexes(X_train)

    for e in range(epochs):
        for batch_count in range(m // batch_size):  # batch
            images_indexes = []
            cluster_images_indexes = []
            batch_ids = np.random.choice(range(0, m), batch_size)
            for i in batch_ids:
                images_indexes += [nn_indexes[i][0]] * (n_neighbors - 1)
                cluster_images_indexes += list(nn_indexes[i][1:])

            images = tf.convert_to_tensor(X_train[images_indexes, ...])
            cluster_images = tf.convert_to_tensor(X_train[cluster_images_indexes, ...])

            with tf.GradientTape() as tape:
                images_pred, cluster_images_pred = clustering_model(images), clustering_model(cluster_images)
                dot_product = tf.reduce_sum(tf.multiply(images_pred, cluster_images_pred), 1)
                consistant_loss = -tf.reduce_mean(tf.math.log(1e-10 + dot_product))

                _ = tf.reduce_mean(images_pred, 0)
                entropy_loss = tf.reduce_sum(_ * tf.math.log(1e-8 + _))

                loss = consistant_loss + lam * entropy_loss

                grads = tape.gradient(loss, clustering_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, clustering_model.trainable_weights))

        if e % 1 == 0:
            # clear_output()
            print(f"epoch {e}, loss: {consistant_loss:.2f} + {lam}*{entropy_loss:.2f}={loss:.3f}")

    if save_path is not None:
        clustering_model.save_weights(save_path)
    return clustering_model

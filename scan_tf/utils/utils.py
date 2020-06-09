import sklearn.neighbors
import numpy as np

class CLusteringNN:

    def __init__(self, backbone, n_neighbors=10):
        self.backbone_cnn = backbone
        self.n_neighbors = n_neighbors
        self.nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        self.embeddings = None

    def fit(self, images):
        embeddings = self.backbone_cnn.predict(images)
        self.nn.fit(embeddings)

    def get_neighbors_indexes(self, images):
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        image_embedding = self.backbone_cnn.predict(images)
        neighbor_indexes = self.nn.kneighbors(image_embedding, return_distance=False)
        return neighbor_indexes

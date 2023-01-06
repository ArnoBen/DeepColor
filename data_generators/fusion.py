import numpy as np
import cv2
from skimage.transform import resize
from tensorflow.keras.applications import InceptionResNetV2


class FusionGenerator:
    def __init__(self):
        self.inception = InceptionResNetV2(weights='imagenet', include_top=True)

    def _create_inception_embedding(self, grayscaled_3d: np.ndarray):
        """
        Infer the given grayscale image to get the 1000 values at the end of a pretrained inception resnet.
        Args:
            grayscaled_3d: grayscale image with 3 equal channels.

        Returns:
            1000 lenght vector of embeddings.
        """
        def resize_gray(x):
            return resize(x, (299, 299, 3), mode='constant')
        grayscaled_3d_resized = np.array([resize_gray(x) for x in grayscaled_3d])
        embed = self.inception.predict(grayscaled_3d_resized, verbose=0)
        # print(embed.min(), embed.max())
        return embed

    @staticmethod
    def preprocessing(image):
        image = np.array(image, dtype=np.uint8)
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) / 255
        return lab_image

    def generator(self, data_generator):
        n = data_generator.samples
        batch_size = data_generator.batch_size
        while True:
            batch = data_generator.next()
            L_batch = batch[..., 0][..., np.newaxis]
            ab_batch = batch[..., 1:]
            grayscale_3d = np.concatenate([L_batch] * 3, axis=-1)
            embed = self._create_inception_embedding(grayscale_3d)
            yield [L_batch, embed], ab_batch

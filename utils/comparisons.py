import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


def compare_output(generator, model, model2=None):
  L_batch, ab_batch = next(generator)
  batch_size = L_batch.shape[0]
  n_imgs = min(batch_size, 6)
  n_variations = 4 if model2 else 3
  fig, axes = plt.subplots(n_variations, n_imgs, figsize=(18, 3 * n_variations))
  
  img_batch = np.concatenate((L_batch, ab_batch), axis=-1)
  img_batch = (img_batch * 255).astype(np.uint8)
  
  ab_pred_batch = model.predict(L_batch, verbose=0)
  img_pred_batch = np.concatenate((L_batch, ab_pred_batch), axis=-1)
  img_pred_batch = (img_pred_batch * 255).astype(np.uint8)

  if model2:
    ab_pred_batch_2 = model2.predict(L_batch, verbose=0)
    img_pred_batch_2 = np.concatenate((L_batch, ab_pred_batch_2), axis=-1)
    img_pred_batch_2 = (img_pred_batch_2 * 255).astype(np.uint8)
  
  for i, samples in enumerate(zip(L_batch, img_batch, img_pred_batch)):
    L, img, img_pred = samples
    axes[0, i].imshow(L[..., 0], cmap='gray')
    axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_LAB2RGB))
    axes[2, i].imshow(cv2.cvtColor(img_pred, cv2.COLOR_LAB2RGB))
    if model2:
      axes[3, i].imshow(cv2.cvtColor(img_pred_batch_2[i], cv2.COLOR_LAB2RGB))
      
    if i == n_imgs - 1:
      plt.tight_layout()
      break

  titles = ['grayscale', 'original', 'pretrained', 'gan']
  for i in range(n_variations):
    axes[i, 0].set_ylabel(titles[i], fontsize=20)

  plt.show()


class GenerateImageCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(GenerateImageCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        compare_output(self.model)
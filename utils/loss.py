from tensorflow.keras.losses import mean_absolute_error
from tensorflow.image import psnr
from tqdm import tqdm
import numpy as np

def get_loss_values(model, generator, valid_data, embed=False):
  mae_list = []
  psrn_list = []
  
  if embed:
    for i, batch in tqdm(enumerate(generator(valid_data)), total=len(valid_data)):
      if i == len(valid_data):
        break
      input, target = batch
      L, embeding = input
      imgs = np.concatenate((L, target), axis=-1) 
      y_pred = model.predict(input, verbose=0)
      mean_mae = np.mean(mean_absolute_error(target, y_pred))
      mae_list.append(mean_mae) 

      imgs_pred = np.concatenate((L, y_pred), axis=-1) 
      mean_psnr = np.mean(psnr(imgs, imgs_pred, max_val=1.0))
      psrn_list.append(mean_psnr)

  else:
    for i, batch in tqdm(enumerate(generator(valid_data)), total=len(valid_data)):
      if i == len(valid_data):
        break
      L, ab = batch
      imgs = np.concatenate((L, ab), axis=-1) 
      y_pred = model.predict(L, verbose=0)
      mean_mae = np.mean(mean_absolute_error(ab, y_pred))
      mae_list.append(mean_mae) 

      imgs_pred = np.concatenate((L, y_pred), axis=-1) 
      mean_psnr = np.mean(psnr(imgs, imgs_pred, max_val=1.0))
      psrn_list.append(mean_psnr)
  
  return np.mean(mae_list), np.mean(psrn_list)
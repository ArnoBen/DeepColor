import numpy as np
import cv2


def decompose(img: np.ndarray):
    # Input: Keeping L channel
    L = img[:, :, 0]
    # Target: Keeping a and b channels
    ab = img[:, :, 1:]
    return L, ab / 127


def decompose_generator(X: np.ndarray, batch_size: int = 1):
    """
    X: Set images, should be array of shape (n, w, h, 3)
    batch_size: Set batch size
    is_val: Is this a validation set ? Necessary to stop the iterator for validation sets.
    """
    i = 0
    n = len(X)
    end_index = n - n % batch_size
    while True:
        L = X[i: i + batch_size][:, :, :, 0][:, :, :, None]
        ab = X[i: i + batch_size][:, :, :, 1:] / 127
        yield L, ab
        i += batch_size
        if i >= end_index:
            i = 0
            

if __name__ == "__main__":
    from utils.conversions import bgr2lab
    # Cheking if decomposition works well
    sample = bgr2lab(cv2.imread('images/00506.jpg'))

    L_input, ab_true = decompose(sample)
    recomposed = np.zeros(sample.shape)
    recomposed[:, :, 0] = L_input
    recomposed[:, :, 1:] = ab_true * 127

    # fig, axes = plt.subplots(1, 2)
    # axes[0].set_title('recomposed')
    # axes[0].imshow(lab2rgb(recomposed))
    # axes[1].set_title('original')
    # axes[1].imshow(lab2rgb(sample))
    assert np.all(recomposed == sample)
    print("The decomposition - recomposition is working")


# # Image transformer
# datagen = ImageDataGenerator(
#         shear_range=0.2,
#         zoom_range=0.2,
#         # rotation_range=20,
#         horizontal_flip=True)

# # Generate training data
# def image_gen(X, batch_size):
#     for batch in datagen.flow(X, batch_size=batch_size):
#         lab_batch = [cv2.cvtColor(img, cv2.COLOR_BGR2Lab) for img in batch]
#         lab_batch = np.array(lab_batch)
#         # Input: Keeping L channel
#         X_batch = lab_batch[:, :, :, 0][:, :, :, np.newaxis]  
#         # Target: Keeping a and b channels
#         Y_batch = lab_batch[:, :, :, 1:]
#         yield X_batch, Y_batch / 128

# # Display some examples of transformation
# fix, axes = plt.subplots(2, 4, figsize=(14, 6))
# for i in range(8):
#     image = cv2.cvtColor(next(datagen.flow(X, batch_size=1))[0].astype(np.uint8), cv2.COLOR_Lab2RGB)
#     axes.flat[i].imshow(image)
# plt.show()
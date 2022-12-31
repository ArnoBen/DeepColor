import numpy as np
import cv2


def decompose(img: np.ndarray):
    """
    Separates the Lab channel ranges and normalizes the ranges:
    L: 0, 100 => -1, 1
    ab: -127, 127 => -1, 1
    """
    # Input: Keeping L channel
    # Keeping the shape as (h, w, 1) because it's consistent with ab (h, w, 2)
    # and is expected as input in the GAN's generator
    L = img[:, :, 0][..., np.newaxis] / 50 - 1
    # Target: Keeping a and b channels
    ab = img[:, :, 1:] / 127
    return L, ab


def recompose(L: np.ndarray, ab: np.ndarray):
    """
    Concatenates the L and ab channel ranges and rescales the value ranges:
    L: -1, 1  => 0, 100
    ab: -1, 1  => -127, 127
    """
    assert L.shape[:2] == ab.shape[:2], f"Different shapes: {L.shape[:2]}, {ab.shape[:2]}"
    image = np.zeros((L.shape[0], L.shape[1], 3))
    image[:, :, 0] = (L[..., 0] + 1) * 50
    image[:, :, 1:] = ab * 127
    return np.round(image).astype(int)


def decompose_generator(X: np.ndarray, batch_size: int = 1, flip_horizontal=True):
    """
    X: Set images, should be array of shape (n, h, w, 3)
    batch_size: Set batch size
    """
    i = 0
    n = len(X)
    end_index = n - n % batch_size
    while True:
        L = X[i: i + batch_size][..., 0] / 50 - 1
        ab = X[i: i + batch_size][..., 1:] / 127
        if flip_horizontal and np.random.randint(0, 2):  ## coinflip
            yield L[:, ::-1, :], ab[:, ::-1, :]
        else:
            yield L, ab
        i += batch_size
        print("\ni : ", i)
        if i >= end_index:
            i = 0


if __name__ == "__main__":
    from conversions import bgr2lab
    # Cheking if decomposition works well
    sample = bgr2lab(cv2.imread('lena-256.jpg'))

    L_input, ab_true = decompose(sample)
    recomposed = recompose(L_input, ab_true)

    assert np.all(recomposed == sample)
    print("The decomposition - recomposition is working")

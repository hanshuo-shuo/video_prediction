from PIL import Image
from tqdm import tqdm
import numpy as np


def main():
    data = np.load("data/mnist_test_seq.npy")
    for i, batch in tqdm(enumerate(data)):
        for j, frame in enumerate(batch):
            imagearray = np.uint8(frame*255)
            data = Image.fromarray(imagearray)
            data.save(f"images/{i}-{j}.png")

    for video in tqdm(range(10000)):
        frames = []
        for frame in range(20):
            frames.append(Image.open(
                f"images/{frame}-{video}.png"))
        frames[0].save(f"gif_real/test_{video}.gif",
                       format='GIF', append_images=frames[1:], save_all=True)

if __name__ == "__main__":
    main()


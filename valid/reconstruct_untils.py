import numpy as np
from matplotlib import pyplot as plt, gridspec


def save_images(images, fig_suffix):
    columns = 4
    rows = (images.shape[0] + 1) // columns
    fig = plt.figure(figsize=(32, (16 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(images[j])
    plt.savefig('images_%s.png' % fig_suffix)


def save_images2(img, fig_suffix):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.savefig('images_%s.png' % fig_suffix)

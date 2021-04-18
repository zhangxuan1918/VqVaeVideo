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
    plt.savefig('video_frames_%s.png' % fig_suffix)
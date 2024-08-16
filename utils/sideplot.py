import cv2
import numpy as np
import matplotlib.pyplot as plt


def side_plot(images, titles, output_file='tokewise_plot.png'):
    fig, axes = plt.subplots(4, 4, figsize=(17, 17))

    axes = axes.flatten() # Flatten the axes array for easier indexing
    cv2.imwrite(f"run/input_1.png", images[0] * 255)
    for i in range(len(images)):
        if i == 0:
            ax = axes[i]
            ax.imshow(images[0])
        
        elif i> 0:
            ax = axes[i]
            im = ax.imshow(images[i], cmap="seismic")
            ax.set_title(titles[i])
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    for i in range(14):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)


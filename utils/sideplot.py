import cv2
import numpy as np
import matplotlib.pyplot as plt


def side_plot(images, titles, output_file='tokewise_plot.png'):
    fig, axes = plt.subplots(7, 4, figsize=(16, 16))
    fig.suptitle("Image Grid", fontsize=16)

    axes = axes.flatten() # Flatten the axes array for easier indexing

    for i in range(0, 28, 2):
        if i == 0:
            ax = axes[i]
            ax.imshow(images[i//2])
        elif i > 0 and (i // 2) < len(images):
            ax = axes[i]
            ax.imshow(overlay_heatmap_array(images[0], images[i//2]), cmap="autumn_r")  # Images[0] is input image.
            ax.set_title(f"Imoposed for {titles[i//2]}")
            ax.axis('off')

            ax = axes[i+1]
            im = ax.imshow(images[i//2], cmap="seismic")
            ax.set_title(titles[i//2])
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(28):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)


def overlay_heatmap_array(image, heatmap, cmap='jet', alpha=0.7):
    """Overlays a heatmap on an image and returns the result as a NumPy array.

  Args:
    image: A NumPy array representing the image.
    heatmap: A NumPy array representing the heatmap data.
    cmap: The colormap to use for the heatmap.
    alpha: The opacity of the heatmap.

  Returns:
    A NumPy array representing the overlaid image.
  """

    # Ensure image and heatmap have the same shape
    if image.shape != heatmap.shape:
        heatmap = cv2.resize(heatmap, image.shape[:2])

    # Normalize heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # # Convert heatmap to RGB
    # cmap = plt.get_cmap(cmap)
    # heatmap_rgb = cmap(heatmap)[:, :, :3]

    # Apply alpha channel
    overlaid_image = image + jet_heatmap * alpha

    # Convert to uint8 if necessary
    if image.dtype == np.uint8:
        overlaid_image = overlaid_image.astype(np.uint8)

    return overlaid_image

import matplotlib.pyplot as plt


def plot_images(x, in_row):
    num = len(x)
    rows = (num // in_row) + (1 if num % in_row else 0) 
    fig, axes = plt.subplots(nrows=rows, ncols=in_row, figsize=(10, 2 * rows))
    axes = axes.flatten()

    for i in range(num):
        axes[i].imshow(x[i].squeeze(), cmap='gray')
        # axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')

    # Hide unused subplots
    for j in range(num, len(axes)):
        axes[j].axis('off')

    plt.show()

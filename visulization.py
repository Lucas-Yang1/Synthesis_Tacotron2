import matplotlib.pyplot as plt
import numpy as np


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8 )
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.savefig(f'archive/IMG/{info}.png')
    plt.close()
    return data

def plot_spectrogram_to_numpy(spectrogram1, spectrogram2, info=None):
    fig, axes = plt.subplots(2,1,figsize=(12, 3))
    im = axes[0].imshow(spectrogram1, aspect="auto", origin="lower",
                   interpolation='none', vmin=-3.5, vmax=3.5)
    plt.colorbar(im, ax=axes[0])
    #plt.title("Target", ax=axes[0])
    axes[0].set(xlabel="Frames", ylabel="Channel", title='Target')#ax=axes[0])

    im = axes[1].imshow(spectrogram2, aspect="auto", origin="lower",
                          interpolation='none', vmin=-3.5, vmax=3.5)
    plt.colorbar(im, ax=axes[1])
    axes[1].set(xlabel="Frames", ylabel="Channel", title='Predict')#ax=axes[0])

    plt.tight_layout()

    fig.canvas.draw()
    plt.savefig(f"./archive/spectro/{info}.png")
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

import matplotlib.pyplot as plt

def plot_spectrogram(spec):

    plt.figure(figsize=(10,4))
    plt.imshow(spec,aspect='auto',origin='lower')
    plt.colorbar()
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.show()

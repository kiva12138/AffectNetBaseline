import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.figure(figsize=(5, 5))
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    plt.plot(0.5, 0.5, 'ob', color='r')
    plt.text(0.5, 0.5, 'Happy', color='r')
    plt.plot([1,-1], [0,0], color='y')
    plt.plot([0,0], [1,-1], color='y')
    plt.gcf().gca().add_artist(plt.Circle((0, 0), 1, color='y', fill=False))

    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title("VA")
    plt.show()
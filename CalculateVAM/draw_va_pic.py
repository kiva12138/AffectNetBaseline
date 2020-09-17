import json
import ast
import io
import matplotlib.pyplot as plt
import PIL.Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor


expressions = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'Non-Face']
colors = ['red', 'green', 'blue', 'purple', 'gray', 'fuchsia', 'cyan', 'darkblue']

if __name__ == "__main__":
    with open('vam.json', 'r') as f:
            data = json.load(f)
    data = ast.literal_eval(data)
    writer = SummaryWriter('./logdir')

    plt.figure(figsize=(5, 5))
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)

    for i in range(8):
        plt.plot(data['valence'][str(i)], data['arousal'][str(i)], 'ob', color=colors[i])
        plt.text(data['valence'][str(i)], data['arousal'][str(i)], expressions[i], color=colors[i])
    plt.plot([1,-1], [0,0], color='yellow')
    plt.plot([0,0], [1,-1], color='yellow')
    plt.gcf().gca().add_artist(plt.Circle((0, 0), 1, color='yellow', fill=False))

    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title("VA")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = ToTensor()(PIL.Image.open(buf))

    writer.add_image('VA', image)
    writer.close()
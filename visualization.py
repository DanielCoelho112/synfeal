from math import pi
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch import nn
import cv2

class Visualizer():

    def __init__(self, title):
       
        # Initial parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title
        self.tensor_to_pil_image = transforms.ToPILImage()

    def draw(self, inputs, labels, outputs):

        # Setup figure
        self.figure = plt.figure(self.title)
        plt.axis('off')
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.set_size_inches(12,7)
        self.figure.tight_layout()
        plt.suptitle(self.title)

        inputs = inputs
        batch_size,_,_,_ = list(inputs.shape)

        if batch_size < 25:
            random_idxs = random.sample(list(range(batch_size)), k=batch_size)
        else:
            random_idxs = random.sample(list(range(batch_size)), k=5*5)
        plt.clf()
        
        for plot_idx, image_idx in enumerate(random_idxs, start=1):
            outputs_np = np.array([output.data.item() for output in outputs[image_idx]])
            labels_np = np.array([label.data.item() for label in labels[image_idx]])
            if all(outputs_np - labels_np) < 0.01:
                color='green'
            else:
                color='red'

            image_t = inputs[image_idx,:,:,:]
            image_pil = self.tensor_to_pil_image(image_t)

            ax = self.figure.add_subplot(5,3,plot_idx) # define a 5 x 5 subplot matrix
            plt.imshow(np.asarray(image_pil))
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_xlabel(f'{np.around(outputs_np,2)}\n{np.around(labels_np,2)}', color=color)

        plt.draw()
        key = plt.waitforbuttonpress(0.05)
        if not plt.fignum_exists(1):
            print('Terminating')
            exit(0)

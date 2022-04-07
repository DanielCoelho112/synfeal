from torchsummary import summary
import os
import yaml
from yaml.loader import SafeLoader
from localbot_localization.src.models.pointnet import PointNet
from localbot_localization.src.models.depthnet import CNNDepth
import torch
from colorama import Fore


def summarizeModel(model, input_example):
    model.cuda()
    summary(model, input_size=input_example.shape)
    model.cpu()
    
    
def resumeTraining(folder_name):
    model_name = [f for f in os.listdir(folder_name) if f.endswith('.pth')][0] # get first in the list of files that have extension .pth
    file_name = f'{folder_name}/config.yaml'

    with open(file_name) as f:
        config = yaml.load(f, Loader=SafeLoader)

    model = eval(config['init_model'])
    model.load_state_dict(torch.load(f'{folder_name}/{model_name}'))

    start_epoch = config['epoch']
    train_losses = config['train_losses']
    test_losses = config['test_losses']
    
    print(f'{Fore.BLUE} Resuming training of model from epoch: {start_epoch} {Fore.RESET}')
    
    return start_epoch, train_losses, test_losses, model


# def viewDatasetImages(dataset):
#         for idx in range(len(train_dataset)):
#             points, depth_image, target_pose = train_dataset[idx]
#             np_depth_image = depth_image.numpy()
#             np_depth_image = np_depth_image.reshape((224, 224))
#             print(np_depth_image.shape)
#             win_name = 'Dataset image ' + str(idx)
#             cv2.imshow(win_name, np_depth_image)
#             cv2.waitKey(0)
#             cv2.destroyWindow(win_name)
#             # TODO keypress of q to finish visualization
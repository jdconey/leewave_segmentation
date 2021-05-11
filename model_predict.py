import os
import numpy as np
import torch
import xarray
from unet2 import UNet
#from model2_dice import CrossEntropy2d
import pickle

from transformations import Compose, Resize, DenseTarget
from transformations import MoveAxis, Normalize01, AlbuSeg2d
import wandb
import matplotlib.pyplot as plt

wandb.init(project="potato")

transforms = Compose([
    # Resize(input_size=(128, 128, 3), target_size=(128, 128)),
    DenseTarget(),
    MoveAxis(),
    Normalize01()
])


input_ID ="/nobackup/mm16jdc/leewave_segmentation/new_data/20210129T1500Z-PT0000H00M-wind_vertical_velocity_at_700hPa.nc"

model_loc = '/nobackup/mm16jdc/model_2021-05-10_14.pt'

print('hello')
model = UNet(n_classes=2, padding=True, up_mode='upsample')
print('hello again')


#criterion
#criterion = torch.nn.CrossEntropyLoss()
#criterion= IoULoss()
#criterion = CrossEntropy2d()
# optimizer
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1,momentum = 0.9)
# trainer

device = torch.device('cpu')

checkpoint = torch.load('/nobackup/mm16jdc/model_2021-04-29.pt',map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = 50
#loss = checkpoint['loss']

#model.eval()


      #  val_accuracies = []



x = xarray.open_dataarray(input_ID)
x = x.values
y = torch.zeros(512,512)
        # Preprocessing
if transforms is not None:
    x,y = transforms(x,y)
                    

x = torch.from_numpy(x)

x = x.unsqueeze(0)  # if torch tensor
x = x.unsqueeze(0)

input = x.to(device)

with torch.no_grad():
        out = model(input)
 #       loss = criterion(out,target)
        #loss = self.criterion(torch.argmax(out, dim=1), target)
  #      loss_value = loss.item()
   #     print(loss_value)
         
         #GET PREDICTED MASK
        pred = torch.argmax(out, dim=1)  # perform argmax to generate 1 channel
        pred = pred.cpu().numpy()  # send to cpu and transform to numpy.ndarray
        pred = np.squeeze(pred)  # remove batch dim and channel dim -> [H, W]

        
#        pickle.dump(out,'/nobackup/mm16jdc/leewave_segmentation/new_data/out.pkl')
        plt.imshow(pred,origin='lower')
        plt.colorbar()
        plt.savefig('/nobackup/mm16jdc/leewave_segmentation/new_data/out.pdf')

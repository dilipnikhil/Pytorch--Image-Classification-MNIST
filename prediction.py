import torchNN
from torchNN import ImageClassifier
import torch
from torch import save,load
from PIL import Image
from torchvision.transforms import ToTensor



clf = ImageClassifier().to("cuda")
with open ('model_state.pt',"rb") as f:
    clf.load_state_dict(load(f))


img = Image.open('five.JPG')
img_tensor = ToTensor()(img).unsqueeze(0).to("cuda")

print(torch.argmax(clf(img_tensor)))

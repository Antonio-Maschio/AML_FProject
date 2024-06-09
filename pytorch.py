from torch.utils.data import Dataset as DS
class ImageDataset(DS):
    def __init__(self, df, image_type='nucleus'):
        from numpy.random import permutation

        if image_type == 'nucleus': self.fnames = df['nucleus_fname']
        else: self.fnames = df['cell_fname']

        self.labels = df['label']

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        from torchvision.io import read_image
        return read_image(self.fnames[idx]).float(), self.labels[idx]

class LoadImageDataset(DS):
    def __init__(self, parent_dir):
        from os import listdir

        control_dir = parent_dir + 'control/'
        drug_dir = parent_dir + 'drug/'

        self.fnames = [control_dir + fname for fname in listdir(control_dir)] + [drug_dir + fname for fname in listdir(drug_dir)]
        self.labels = [0 for _ in listdir(control_dir)] + [1 for _ in listdir(drug_dir)]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        from torchvision.io import read_image
        return read_image(self.fnames[idx]).float(), self.labels[idx]
    
### Pretrained Models
    
from torchvision.models import resnet34
from torch.nn import Module, Sequential, Flatten, Linear, ReLU

class Resnet34L(Module):
    def __init__(self, n_layers:int=3, n_nodes:int=512):
        super().__init__()

        rn34 = resnet34(pretrained=True)

        model = Sequential(*list(rn34.children())[:7])
        for p in model.parameters():
            p.requires_grad = False

        model.append(Flatten())

        for i in range(n_layers):
            if i == 0:
                model.append(Linear(50_176, n_nodes))
                model.append(ReLU())
            elif i == n_layers-1:
                model.append(Linear(n_nodes, 2))
                model.append(ReLU())
            else:
                model.append(Linear(n_nodes, n_nodes))

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def getSummary(self):
        from torchsummary import summary
        return summary(self, (3, 224, 224))
    
class Resnet34M(Module):
    def __init__(self, n_layers:int=3, n_nodes:int=512):
        super().__init__()

        rn34 = resnet34(pretrained=True)

        model = Sequential(*list(rn34.children())[:8])
        for p in model.parameters():
            p.requires_grad = False

        model.append(Flatten())

        for i in range(n_layers):
            if i == 0:
                model.append(Linear(25_088, n_nodes))
                model.append(ReLU())
            elif i == n_layers-1:
                model.append(Linear(n_nodes, 2))
                model.append(ReLU())
            else:
                model.append(Linear(n_nodes, n_nodes))

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def getSummary(self):
        from torchsummary import summary
        return summary(self, (3, 224, 224))
    
class Resnet34S(Module):
    def __init__(self, n_layers:int=3, n_nodes:int=512):
        super().__init__()

        rn34 = resnet34(pretrained=True)

        model = Sequential(*list(rn34.children())[:9])
        for p in model.parameters():
            p.requires_grad = False

        model.append(Flatten())

        for i in range(n_layers):
            if i == 0:
                model.append(Linear(512, n_nodes))
                model.append(ReLU())
            elif i == n_layers-1:
                model.append(Linear(n_nodes, 2))
                model.append(ReLU())
            else:
                model.append(Linear(n_nodes, n_nodes))

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def getSummary(self):
        from torchsummary import summary
        return summary(self, (3, 224, 224))
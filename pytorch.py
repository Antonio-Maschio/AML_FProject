from torch.utils.data import Dataset
class ImageDataset(Dataset):
    def __init__(self, df, image_type='nucleus'):
        from numpy.random import permutation

        data = df.reindex(permutation(df.index))

        if image_type == 'nucleus': self.fnames = data['nucleus_fname']
        else: self.fnames = data['cell_fname']

        self.labels = data['label']

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        from torchvision.io import read_image
        return read_image(self.fnames[idx]), self.labels[idx]

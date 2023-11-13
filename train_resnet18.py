
import torch
from torch import nn
from torchvision import models

from torch.utils.data import DataLoader, Dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class HiddenDataset(Dataset):
    '''The hidden dataset.'''
    def __init__(self, root_path, split='train'):
        super().__init__()
        self.examples = []

        df = pd.read_csv(f'/kaggle/input/neurips-2023-machine-unlearning/{split}.csv')
        df['image_path'] = df['image_id'].apply(
            lambda x: os.path.join('/kaggle/input/neurips-2023-machine-unlearning/', 'images', x.split('-')[0], x.split('-')[1] + '.png'))
        df = df.sort_values(by='image_path')
        df.apply(lambda row: self.examples.append(load_example(row)), axis=1)
        if len(self.examples) == 0:
            raise ValueError('No examples.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image = example['image']
        image = image.to(torch.float32)
        example['image'] = image
        return example

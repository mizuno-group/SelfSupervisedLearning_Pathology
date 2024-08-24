# -*- coding: utf-8 -*-
"""
# Image Analyzer

@author: Katsuhisa MORITA
"""

from .imageprocessor import ImageProcessor
from .model import FindingClassifier

class Analyzer()
    def __init__(self):
        # analyzing class
        self.FindingClassifier=FindingClassifier()
        self.ImageProcessor=ImageProcessor()
        # data
        self.image=None
        self.mask=None
        self.probabilities=None

    def load_model(self, dir_featurize_model="model.pt", dir_classification_models="folder or ", style="dict")
        self.FindingClassifier.load_featurize_model(dir_model=dir_featurize_model)
        self.FindingClassifier.load_classification_models(dir_models=dir_classification_models. style=style)

    def analyze(self, filein, size=448):
        self.image=self.ImageProcessor.load_image(filein)
        locatinos, data = self.ImageProcessor.topatch(size=size, sort_block=True)
        data_loader=self.prepare_dataloader()
        lst_names, preds =self.classify(data_loader, num_pool=int(size/224))


# DataLoader
class PatchDataset(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                data=None,
                transform=None,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # set
        self.data=data
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

def prepare_dataset(data=None, batch_size:int=128):
    """
    data preparation
    """
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    # data
    dataset = PatchDataset(
        data=data,
        transform=data_transform,
        num_patch=num_patch,
        )
    # to loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        sampler=None,
        collate_fn=None
        )
    return data_loader

def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
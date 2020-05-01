import os.path as osp
import scipy.io as io
from torch.utils.data import Dataset
from PIL import Image


__all__ = ['CUB200', 'Cars196', 'StanfordOnlineProducts']


class CUB200(Dataset):
    r"""Caltech-UCSD Birds-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200.html> Dataset.

    Arguments:
        root_dir (str): Root directory of the dataset. (default: None)
        subset (str): Subset of the dataset. Options include 'train',
            'test', and 'full'. (default: 'train')
        split_method (str): The way to split the dataset. Options
            include 'classwise' and 'default'. If 'classwise' is chozen,
            images belonging to the first/last half number of classes of
            the dataset will be considered as the 'train'/'test' subset,
            respectively; Otherwise, if 'default' is chozen, the 'train'
            and 'test' subset will be splitted according to the
            'train_test_split.txt' file included in the dataset.
            (default: 'classwise')
        transforms (object): Data transforms applied to the image.
            (default: None)
    """
    def __init__(self, root_dir=None, subset='train',
                 split_method='classwise', transforms=None):
        if root_dir is None:
            root_dir = osp.expanduser('~/data/CUB_200_2011')
        assert osp.exists(root_dir)
        assert subset in ['train', 'test', 'full']
        assert split_method in ['default', 'classwise']
        self.root_dir = root_dir
        self.subset = subset
        self.split_method = split_method
        self.transforms = transforms

        # load image ID-path dictionary
        with open(osp.join(root_dir, 'images.txt')) as f:
            imgs = f.read().strip().split('\n')
            imgs = [t.split() for t in imgs]
        id2path = {k: osp.join(root_dir, 'images', v) for k, v in imgs}

        # load image ID-label dictionary
        with open(osp.join(root_dir, 'image_class_labels.txt')) as f:
            labels = f.read().strip().split('\n')
            labels = [t.split() for t in labels]
        id2label = {k: int(v) - 1 for k, v in labels}

        # load label-classname dictionary
        with open(osp.join(root_dir, 'classes.txt')) as f:
            classes = f.read().strip().split('\n')
            classes = [t.split() for t in classes]
        label2class = {int(k) - 1: v for k, v in classes}

        # split dataset according to split_method
        img_ids = [t[0] for t in imgs]
        if subset != 'full':
            if split_method == 'default':
                with open(osp.join(
                    root_dir, 'train_test_split.txt')) as f:
                    splits = f.read().strip().split('\n')
                    splits = [t.split() for t in splits]
                flag = '0' if subset == 'train' else '1'
                img_ids = [t[0] for t in splits if t[1] == flag]
            else:  # class-wise splitting
                if subset == 'train':
                    img_ids = [k for k in img_ids if id2label[k] < 100]
                else:  # test subset
                    img_ids = [k for k in img_ids if id2label[k] >= 100]
        
        # collect image files
        self.img_files = [id2path[k] for k in img_ids]

        # collect labels
        labels = [id2label[k] for k in img_ids]
        names = [label2class[k] for k in labels]
        self.class_names = sorted(set(names))
        self.labels = [self.class_names.index(k) for k in names]
    
    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.img_files)
    
    @property
    def CLASSES(self):
        return self.class_names


class Cars196(Dataset):
    r"""Cars-196 <http://ai.stanford.edu/~jkrause/cars/car_dataset.html> Dataset.

    Arguments:
        root_dir (str): Root directory of the dataset. (default: None)
        subset (str): Subset of the dataset. Options include 'train',
            'test', and 'full'. (default: 'train')
        split_method (str): The way to split the dataset. Options
            include 'classwise' and 'default'. If 'classwise' is chozen,
            images belonging to the first/last half number of classes of
            the dataset will be considered as the 'train'/'test' subset,
            respectively; Otherwise, if 'default' is chozen, the 'train'
            and 'test' subset will be splitted according to the
            'cars_train_annos.mat' and 'cars_test_annos_withlabels.mat'
            files included in the dataset. (default: 'classwise')
        transforms (object): Data transforms applied to the image.
            (default: None)
    """
    def __init__(self, root_dir=None, subset='full',
                 split_method='classwise', transforms=None):
        if root_dir is None:
            root_dir = osp.expanduser('~/data/cars196')
        assert osp.exists(root_dir)
        assert subset in ['train', 'test', 'full']
        assert split_method in ['default', 'classwise']
        self.root_dir = root_dir
        self.subset = subset
        self.transforms = transforms

        # load training image paths and annotations
        train_annos = io.loadmat(osp.join(
            root_dir,
            'devkit/cars_train_annos.mat'))['annotations']
        train_imgs = [osp.join(root_dir, 'cars_train', t.item())
                      for t in train_annos['fname'][0]]
        train_labels = [t.item() - 1 for t in train_annos['class'][0]]

        # load test image paths and annotations
        test_annos = io.loadmat(osp.join(
            root_dir,
            'devkit/cars_test_annos_withlabels.mat'))['annotations']
        test_imgs = [osp.join(root_dir, 'cars_test', t.item())
                     for t in test_annos['fname'][0]]
        test_labels = [t.item() - 1 for t in test_annos['class'][0]]

        # load class names
        meta = io.loadmat(
            osp.join(root_dir, 'devkit/cars_meta.mat'))
        class_names = [t.item() for t in meta['class_names'][0]]

        # split dataset according to split_method
        if split_method == 'default':
            if subset == 'train':
                self.img_files = train_imgs
                self.labels = train_labels
            elif subset == 'test':
                self.img_files = test_imgs
                self.labels = test_labels
            elif subset == 'full':
                self.img_files = train_imgs + test_imgs
                self.labels = train_labels + test_labels
            self.class_names = class_names
        elif split_method == 'classwise':
            img_files = train_imgs + test_imgs
            labels = train_labels + test_labels
            num_imgs = len(img_files)
            # separate training and test subsets according to classes
            if subset == 'train':
                self.img_files = [
                    img_files[i] for i in range(num_imgs)
                    if labels[i] < 98]
                self.labels = [
                    labels[i] for i in range(num_imgs)
                    if labels[i] < 98]
                self.class_names = class_names[:98]
            elif subset == 'test':
                self.img_files = [
                    img_files[i] for i in range(num_imgs)
                    if labels[i] >= 98]
                self.labels = [
                    labels[i] - 98 for i in range(num_imgs)
                    if labels[i] >= 98]
                self.class_names = class_names[98:]
            elif subset == 'full':
                self.img_files = img_files
                self.labels = labels
                self.class_names = class_names
    
    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.img_files)
    
    @property
    def CLASSES(self):
        return self.class_names


class StanfordOnlineProducts(Dataset):
    r"""Stanford Online Products <https://cvgl.stanford.edu/projects/lifted_struct/> Dataset.

    Arguments:
        root_dir (str): Root directory of the dataset. (default: None)
        subset (str): Subset of the dataset. Options include 'train',
            'test', and 'full'. (default: 'train')
        split_method (str): The way to split the dataset. Only
            'classwise' is available (since the default splitting is
            classwise already). Images belonging to the first/last half
            number of classes of the dataset will be considered as the
            'train'/'test' subset, respectively. (default: 'classwise')
        transforms (object): Data transforms applied to the image.
            (default: None)
    """
    def __init__(self, root_dir=None, subset='full',
                 split_method='classwise', transforms=None):
        if root_dir is None:
            root_dir = osp.expanduser('~/data/Stanford_Online_Products')
        assert osp.exists(root_dir)
        assert subset in ['train', 'test', 'full']
        assert split_method == 'classwise', \
            'StanfordOnlineProducts only supports classwise splitting'
        self.root_dir = root_dir
        self.subset = subset
        self.transforms = transforms

        # subset splitting file
        if subset == 'train':
            filename = 'Ebay_train.txt'
        elif subset == 'test':
            filename = 'Ebay_test.txt'
        elif subset == 'full':
            filename = 'Ebay_info.txt'
        
        # collect image files and labels for this subset
        with open(osp.join(root_dir, filename)) as f:
            splits = f.read().strip().split('\n')[1:]
            splits = [t.split() for t in splits]
        self.img_files = [osp.join(root_dir, t[3]) for t in splits]
        labels = [int(t[1]) - 1 for t in splits]

        # collect class names
        with open(osp.join(root_dir, 'Ebay_info.txt')) as f:
            items = f.read().strip().split('\n')[1:]
            items = [t.split() for t in items]
        class_dict = {}
        for item in items:
            img_id, cls_id, sup_id, path = item
            name = '{}_{}_{}'.format(
                path[:path.find('_final')], sup_id, cls_id)
            class_dict[int(cls_id) - 1] = name
        class_names = [class_dict[k] for k in sorted(class_dict.keys())]

        # make labels continuous
        names = [class_names[k] for k in labels]
        self.class_names = sorted(set(names))
        self.labels = [self.class_names.index(k) for k in names]

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        label = self.labels[index]
        return img, label
    
    def __len__(self):
        return len(self.img_files)
    
    @property
    def CLASSES(self):
        return self.class_names

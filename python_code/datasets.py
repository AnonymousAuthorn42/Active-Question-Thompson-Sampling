import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights


class ToyDataset(Dataset):
    def __init__(self, dataset_name, data_path=None, train_bool=True ):
        self.train = train_bool
        self.dataset_name = dataset_name
        self.img_dim = (1,2)
        
        
    def create_data_TOY(self, size=20000):
        n_x = max([k if self.n_label%k== 0 else 0 for k in range(2,int(np.sqrt(self.n_label))+1)])
        n_y = int(self.n_label/n_x)
        x = np.arange(n_x)
        y = np.arange(n_y)
        self.n_per_label = np.floor(np.random.normal(size/self.n_label,3,self.n_label)).astype('int')
        self.size=int(np.sum(self.n_per_label))
        for label , tot in enumerate(self.n_per_label):
            coordx, coordy = label%n_x, label//n_x
            Y = np.zeros((int(tot),self.n_label))
            Y[:,label]+=1
            if label==0:
                self.data=np.random.uniform(-1/2,1/2,(int(tot),2)) +np.array((x[coordx],y[coordy]))
                self.labels=Y
                self.targets=np.zeros(int(tot)) #for plots or label information
            else:
                self.data=np.concatenate((self.data , np.random.uniform(-1/2,1/2,(int(tot),2)) +np.array((x[coordx],y[coordy]))))
                self.labels = np.concatenate((self.labels , Y))
                self.targets = np.concatenate((self.targets, np.zeros(int(tot))+label))

        
    def create_data_TOY2(self, size=20000):
        n_x = max([k if self.n_label%k== 0 else 0 for k in range(2,int(np.sqrt(self.n_label))+1)])
        n_y = int(self.n_label/n_x)
        x = np.arange(n_x)
        y = np.arange(n_y)
        self.n_per_label = np.floor(np.random.normal(size/self.n_label,3,self.n_label)).astype('int')
        self.size=np.sum(self.n_per_label)
        for label , tot in enumerate(self.n_per_label):
            coordx, coordy = label%n_x, label//n_x
            Y = np.zeros((tot,self.n_label))
            Y[:,label]+=1
            if label==0:
                self.data=np.random.normal(0,1/4,2*tot).reshape(tot,2) +np.array((x[coordx],y[coordy]))
                self.labels=Y
                self.targets=np.zeros(tot) #for plots or label information
            else:
                self.data=np.concatenate((self.data , np.random.normal(0,1/4,2*tot).reshape(tot,2) +np.array((x[coordx],y[coordy]))))
                self.labels = np.concatenate((self.labels , Y))
                self.targets = np.concatenate((self.targets, np.zeros(tot)+label))

        
    def create_data(self, size = 20000, n_label = 10) :
        self.n_label = n_label
        self.class_to_idx = {f'class_{k}' : k for k in range(self.n_label) }
        if self.dataset_name == 'TOY' :
            self.create_data_TOY(size)
        elif self.dataset_name == 'TOY2' :
            self.create_data_TOY2(size)
        # random permutation
        p = np.random.permutation(np.arange(self.size))
        self.data = self.data[p]   # for plots
        self.labels = self.labels[p]
        self.targets = self.targets[p]
        # normalise self.data for trainable uses
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.data)
        self.data_norm = scaler.transform(self.data)
        self.data_norm = torch.tensor(self.data_norm).float().unsqueeze(1)
        self.labels = torch.tensor(self.labels)
        self.targets = torch.tensor(self.targets).int()
        self.scaler = scaler                  #to inverse_transform if needed    
        #self.partial_labels = torch.ones_like(self.labels)
    
    #def __getitem__(self,idx):
    #    return self.data_norm[idx],self.partial_labels[idx], idx
    
    def __getitem__(self,idx):
        return self.data_norm[idx],self.targets[idx].item() #, idx
    
    #def update_label(self, idx, new_label):
    #    self.partial_labels[idx] = new_label
    
    def __len__(self):
        return self.size

    def show(self):
        colors = ['red','blue','green','black','deeppink','grey','purple','cyan','orange','darkblue']
        colors = np.array(colors)
        for k in range(10):
            pos = np.where(self.targets==k)[0]
            plt.scatter(self.data[pos,0],self.data[pos,1], color=colors[k],s=0.5, label=f'label {k+1}')
        plt.title('Toy Dataset visualization')
        plt.xlim([-1,2])
        plt.ylim([-2,5])
        plt.legend(ncol=5, fontsize=8, markerscale=5)
    
class CirDataset(Dataset) :
    def __init__(self, num_classes = 10, train_bool = True, **kwargs):
        self.train = train_bool
        self.dataset_name = 'Circles'
        self.img_dim = (1,2)
        self.num_classes = num_classes
        try :
            self.size = kwargs['size']
            if self.size is None : self.size = 1000*num_classes if train_bool else 100*num_classes
        except :
            self.size = 1000*num_classes if train_bool else 100*num_classes
        self.create_data()
        
    def create_data(self, size = None):
        self.class_to_idx = {f'class_{k}' : k for k in range(self.num_classes) }
        if size is not None:
            self.size = size
            
        elt_per_class = self.size //self.num_classes
        
        distances = np.arange(self.num_classes)
        variances = np.eye(self.num_classes) /10
        
        r = np.random.multivariate_normal(distances, variances ,elt_per_class)
        theta = np.random.uniform(0, 2*np.pi, (elt_per_class,self.num_classes))
        x, y = r * np.cos(theta), r* np.sin(theta)
        self.data = np.vstack( [np.hstack([x[:,k:k+1],y[:,k:k+1]]) for k in range(self.num_classes)])

        self.targets = np.arange(self.num_classes).repeat(elt_per_class)

        self.size = self.data.shape[0]

        self.labels = np.zeros((self.size, self.num_classes))
        for k in range(self.size):
            self.labels[k,self.targets[k]] = 1.

        p = np.random.permutation(np.arange(self.size))
        self.data = self.data[p]   # for plots
        self.labels = self.labels[p]
        self.targets = self.targets[p]
        # normalise self.data for trainable uses
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.data)
        self.data_norm = scaler.transform(self.data)
        self.data_norm = torch.tensor(self.data_norm).float().unsqueeze(1)
        self.labels = torch.tensor(self.labels)
        self.targets = torch.tensor(self.targets).int()
        self.scaler = scaler
        #self.partial_labels= torch.ones_like(self.labels)
        
    #def __getitem__(self,idx):
    #    return self.data_norm[idx],self.partial_labels[idx], idx
    
    def __getitem__(self,idx):
        return self.data_norm[idx],self.targets[idx].item() #, idx
    
    #def update_label(self, idx, new_label):
    #    self.partial_labels[idx] = new_label
    
    def __len__(self):
        return self.size

    def show(self):
        #colors = ['red','blue','green','black','deeppink','grey','purple','cyan','orange','darkblue']
        #colors = np.array(colors)
        for k in range(self.num_classes):
            pos = np.where(self.targets==k)[0]
            plt.scatter(self.data[pos,0],self.data[pos,1], s=0.5, label=f'label {k+1}')
        plt.title('Circle Dataset visualization')
        if self.num_classes <11 :
            plt.legend(ncol=5, fontsize=8, markerscale=5)
        plt.show()
   
        

class MyDataset(Dataset):
    def __init__(self, dataset_name, data_path, train_bool, preprocess, **kwargs):
        self.dataset_name = dataset_name
        if dataset_name == 'MNIST':
            self.data = datasets.MNIST(root=data_path,
                                        download=False,
                                        train=train_bool,
                                        transform=transforms.ToTensor() )
        elif dataset_name == 'FashionMNIST':
            self.data = datasets.FashionMNIST(root=data_path,
                                        download=True,
                                        train=train_bool,
                                        transform=transforms.ToTensor() )
        # no support for CIFAR 10 yet for questions
        elif dataset_name == 'CIFAR10':
            self.data = datasets.CIFAR10(root = data_path,
                                        download = True,
                                        train = train_bool,
                                        transform = transforms.ToTensor()
                                              )
        elif dataset_name == 'CIFAR100' :
            self.data = datasets.CIFAR100(root=data_path,
                                        download=True,
                                        train=train_bool,
                                        transform=transforms.ToTensor()
                                              )
        elif dataset_name == 'tiny-imagenet-200' :
            data_path2 = os.path.join(data_path, dataset_name, 'train' if train_bool else 'val')
            if not os.path.exists(data_path2) :
                try :
                    prepare_tinyimagenet200_folder(data_path)
                except :
                    raise AssertionError('please download the dataset at http://cs231n.stanford.edu/tiny-imagenet-200.zip')
            self.data = datasets.ImageFolder(data_path2, transform=transforms.ToTensor())

        
        elif dataset_name == 'Circles' :
            try :
                size = kwargs['size']
            except :
                size = None
            try :
                num_classes = kwargs['num_classes']
            except :
                num_classes = 10
            self.data = CirDataset(num_classes=num_classes, train_bool = train_bool, size = size)
            
        elif 'TOY' in dataset_name :
            try :
                size = kwargs['size']
            except :
                size = 60000 if train_bool else 10000
            self.data = ToyDataset(dataset_name, train_bool= train_bool)
            self.data.create_data( size = size )        
        
        self.class_to_idx = self.data.class_to_idx
        self.num_classes = len(self.class_to_idx.keys())        
        self.partial_labels = torch.ones((len(self.data),self.num_classes)).float()
        
        self.labels = torch.zeros_like(self.partial_labels).float()
        for idx, (_,lab) in enumerate(self.data):
            self.labels[idx,lab] = 1.
        self.inputs = torch.stack([sample for _,(sample,_) in enumerate(self.data)])
        
        self.preprocess = preprocess
        self.img_dim = self.__getitem__(0)[0].shape

        if 'device' in kwargs.keys() :
            device = kwargs['device']
            self.inputs = self.inputs.to(device)
            self.partial_labels = self.partial_labels.to(device)
            self.labels = self.labels.to(device)    
        
        

    def __getitem__(self, idx) :
        return self.preprocess(self.inputs[idx]), self.partial_labels[idx], idx
        
    def update_label(self, idx, new_label):
        self.partial_labels[idx]= new_label
    
    def true_labels(self, indices = None ):
        if indices is None :
            self.partial_labels = self.labels
        else :
            self.partial_labels[indices] = self.labels[indices]
            
    def set_val_indices(self, val_set) :
        self.val_indices = val_set.indices
        self.true_labels(self.val_indices)
        
                
    def train_indices(self) :
        indices = torch.where(self.partial_labels.cpu().sum(-1)<self.num_classes)[0].tolist()
        return list(set(indices)-set(self.val_indices))
    
    def classified_indices(self) :
        indices = torch.where(self.partial_labels.sum(-1)==1)[0].tolist()
        return list(set(indices)-set(self.val_indices))
    
    def remaining_samples(self) :
        return torch.where(self.partial_labels.cpu().sum(-1)>1)[0].tolist()

    def __len__(self):
        return len(self.data)



def train_val_split(dataset, val_split= 0.05):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    return train_set, val_set

        
def preprocess_data(net_name, resnet_type, dataset_name):
    means = {'tiny-imagenet-200' : (0.4802, 0.4481, 0.3976),
                 'CIFAR100' : (0.5071, 0.4866, 0.4409),
                  'CIFAR10' : (0.4914, 0.4822, 0.4465),
                   'FashionMNIST' : (0.1307,),
                    'MNIST' : (0.1307,) }[dataset_name]
    stds = {'tiny-imagenet-200' : (0.2770, 0.2691, 0.2821),
                'CIFAR100' : (0.2673, 0.2564, 0.2762),
                 'CIFAR10' : (0.2470, 0.2435, 0.2616),
                   'FashionMNIST' :  (0.3081,),
                    'MNIST' : (0.3081,) }[dataset_name]
    if net_name == 'resnet' :
        if resnet_type == 18 :
            preprocess = ResNet18_Weights.DEFAULT.transforms(antialias=True)
        elif resnet_type == 34 :
            preprocess = ResNet34_Weights.DEFAULT.transforms(antialias=True)
        elif resnet_type == 50:
            preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)
    elif 'MNIST' not in dataset_name and net_name not in ['convnet', 'small_VGG11','small_VGG16']:
        preprocess = ResNet18_Weights.DEFAULT.transforms(antialias=True)
        preprocess.mean = means
        preprocess.std = stds
    else :
        preprocess = transforms.Normalize(means,stds)
    return preprocess


def prepare_tinyimagenet200_folder(data_path) :
    #------------------------------------------------------------------------------
    # please first download tiny-imagenet-200 at link 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    #------------------------------------------------------------------------------
    dataset='tiny-imagenet-200'
    train_doc = os.path.join(data_path, dataset, 'train')
    val_doc = os.path.join(data_path, dataset, 'val')
    anno_file = 'val_annotations.txt'

    boolean = os.path.exists(os.path.join(val_doc, anno_file))
    if boolean :
        with open( os.path.join(val_doc, anno_file),'r') as f:
            txt = f.readlines()
    
        val_img_dict = {}
        for line in txt:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        f.close()
    
        with open(os.path.join(data_path,dataset,'words.txt'),'r') as f:
            label_txt = f.readlines()
    
        f.close()
        label_dict = {}
        for line in label_txt:
            words = line.split('\t')
            label_dict[words[0]] = words[1].split(',')[0].split('\n')[0]
            
        labels = {}
        for doc in os.listdir(train_doc):
            if os.path.isdir(os.path.join(train_doc,doc)) :
                labels[doc] = label_dict[doc]

    
        # reorder train_doc as train/label/img_k.JPEG
        for directory in os.listdir(train_doc):
            path = os.path.join(train_doc, directory)
            image_path = os.path.join(path,'images')
            if os.path.isdir(path) :
                for file_name in os.listdir(path):
                    #move boxes annotations to data_path/dataset/train_annotations
                    if 'boxes' in file_name :
                        old_name = os.path.join(path,file_name)
                        new_name = os.path.join(os.path.join(data_path,dataset,'train_annotations', f'{labels[directory]}_boxes.txt'))
                        if not os.path.exists( os.path.join(data_path,dataset,'train_annotations') ):
                            os.makedirs( os.path.join(data_path,dataset,'train_annotations') )
                        os.rename(old_name, new_name)
                #rename images img_k.JPEG and move it to data_path/dataset/train/directory
                for img in os.listdir(image_path):
                    if '.JPEG' in img :
                        k = img.split('_')[1]
                        k = k.split('.')[0]
                        new_name = os.path.join(path,f'img_{k}.JPEG')
                        old_name = os.path.join(image_path, img)
                        os.rename(old_name,new_name)
                os.rmdir(os.path.join(train_doc,directory,'images'))
                #rename directory name as its label  
                os.rename( path , os.path.join(train_doc,labels[directory]) )
                
        #reorder validation set directory
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(val_doc, labels[folder]))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_doc,'images', img)):
                os.rename(os.path.join(val_doc,'images', img), os.path.join(newpath, img))
        #delete empty image directory        
        os.rmdir(os.path.join(val_doc,'images'))
        #move val annotations
        os.rename(os.path.join(val_doc,'val_annotations.txt'), os.path.join(data_path, dataset, 'val_annotations.txt'))
    return boolean

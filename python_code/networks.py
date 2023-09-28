import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pickle

class Trainer : 
    def __init__(self,model, dataloader, optimizer, loss, data_dir, gpu_id):
        self.optimizer = optimizer
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = model.to(gpu_id)
        self.model = DDP( self.model, device_ids = [gpu_id] )
        self.loader = dataloader
        self.loss = loss
        self.gpu_id = gpu_id
        self.data_dir = data_dir
    
    def run_batch(self, inputs, target) :
        self.optimizer.zero_grad()
        outputs,_ = self.model(inputs)
        loss = self.loss(outputs, target)
        loss.backward()
        self.optimizer.step()
        
    def run_epoch(self, epoch) :
        self.loader.sampler.set_epoch(epoch)
        b_s  = len(next(iter(self.loader))[0])
        for inputs, targets, idx in self.loader :
            inputs = inputs.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self.run_batch(inputs, targets)
            
    def train(self, max_epochs):
        self.model.train()
        for epoch in range(max_epochs):
            self.run_epoch(epoch)

    def eval(self, loader, style):
        self.model.eval()
        logits, embs, idxs = [], [], []
        val_loss = torch.Tensor([0.]).to(self.gpu_id)
        val_acc = torch.Tensor([0.]).to(self.gpu_id)
        with torch.no_grad():
            for inputs, targets, idx in loader :
                output, emb = self.model( inputs.to(self.gpu_id, non_blocking = True) )
                targets = targets.to(self.gpu_id, non_blocking = True)
                loss = self.loss(output, targets)
                val_loss += loss * len(inputs) / (len(loader.dataset))
                val_acc += (output.argmax(-1) == targets.argmax(-1)).sum() / len(loader.dataset)
                logits.append(output)
                embs.append(emb)
                idxs.append(idx.to(self.gpu_id, non_blocking = True))
        logits = torch.vstack(logits)
        embs = torch.vstack(embs)
        idxs = torch.hstack(idxs)
        logit_list, emb_list, idx_list = [torch.zeros_like(logits) for i in range(2)], [torch.zeros_like(embs) for i in range(2)], [torch.zeros_like(idxs) for i in range(2)]
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
        dist.all_gather(logit_list, logits)
        dist.all_gather(emb_list, embs)
        dist.all_gather(idx_list, idxs)
        if self.gpu_id == 0 :
            if style in ['validation', 'inputs'] :
                elts = {'prob' : nn.functional.softmax(torch.vstack(logit_list), dim=-1),
                        'emb' : torch.vstack(emb_list),
                        'idx' : torch.hstack(idx_list),
                        'loss' : val_loss,
                        'acc' : val_acc}
            elif style =='test' :
                elts = {'test_acc' : val_acc,
                        'test_loss' : val_loss}
            with open(os.path.join(self.data_dir,f'{style}.pkl'),'wb') as f:
                pickle.dump(elts, f)






class mlpMod(nn.Module):
    def __init__(self, dim=(28, 28), nclasses=10, embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.fc = nn.Linear(embSize, nclasses)

    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.fc(emb)
        return out, emb

    def get_embedding_dim(self):
        return self.embSize


class ConvNet(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10, img_dim = (1,28,28)):
        super().__init__()
        self.emb_size = 256
        self.m = 4 if img_dim ==(1,28,28) else 5
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*self.m**2, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = t.view(-1, 12*self.m**2)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        emb = F.relu(t)
        t = self.out(emb)
        return t, emb
    
    def get_embedding_dim(self):
        return self.embSize

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=1000):
        super(VGG, self).__init__()
        self.features = self.make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.num_classes = num_classes
        
        if self.num_classes==1000 :
            self.penultimate_dim = 4096
        else:
            self.penultimate_dim = 512

        #Describe model with source code link
        self.description = "VGG16 model loaded from VAAL source code with penultimate dim as {}".format(self.penultimate_dim)

        self.source_link = "https://github.com/sinhasam/vaal/blob/master/vgg.py"
        
        self.penultimate_act = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.penultimate_dim),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.penultimate_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.penultimate_act(x)
        x = self.classifier(emb)
        return x, emb

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return self.penultimate_dim
    
    



class small_VGG(nn.Module):
    def __init__(self, vgg_name = 'small_VGG11', num_classes = 10):
        super(small_VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name.split('_')[-1]])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        emb = out.view(out.size(0), -1)
        out = self.classifier(emb)
        return out, emb

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return 512
 
 



class LeNet5(nn.Module):
    def __init__(self, num_classes=10, grayscale=False, embSize=84):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes
        self.embSize = embSize

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(),  #ReLU instead of Tanh improves performance and is computationaly efficient
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, self.embSize),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.embSize, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        emb = self.classifier(x)
        logits = self.fc(emb)
        return logits, emb
    
    def get_embedding_dim(self):
        return self.embSize








def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()



    
def accuracy(output, target, ks=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(ks)
        batch_size = target.size(0)
        # Get the class index of the top <maxk> scores for each element of the minibatch
        _, pred_ = output.topk(maxk, 1, True, True)
        pred = pred_.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    
        res = []
        for k in ks:
            correct_k = correct[:k].float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        res = torch.vstack(res).sum(-1)
        return res, pred_


def hierarchical_accuracy(output, target, lvls, Q) :
    """
    Parameters
    ----------
    output : torch.Tensor - output distribution from a set of examples.
    target : torch.Tensor - onehot encoding correspond to the set of examples.
    lvls : torch.Tensor - encoding the lvls in the hierarchy - take it from Partial_HierLoss.lvls
    Q : encode the subsets of atomic label for each node in the hierarchy - take it from Questions.Q

    Returns
    -------
    corrects : accuracy at each lvl of the hierarchy (from lvl1 to the leaves).

    """
    with torch.no_grad():
        probs = output @ Q[1:].T
        pos_labels = target @ Q[1:].T
        pos_labels[pos_labels >0] =1.
        num_classes = target.shape[-1]
        hier_pos = pos_labels.unsqueeze(1) * lvls[1:,1:]
        hier_probs = probs.unsqueeze(1) * lvls[1:,1:]
        positivs = hier_pos.argmax(-1) == hier_probs.argmax(-1)
        corrects = positivs.sum(0) /target.shape[0]
    return corrects      
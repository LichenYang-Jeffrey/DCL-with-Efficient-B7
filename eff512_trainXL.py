import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from efficientnet_pytorch import EfficientNet


DIR_INPUT='../../data1/tangyuhao'
PRETRAINED_FILE="./resnet18-5c106cde.pth"
MODELPATH="./models_eff512_XLData"
LOGFILE="./models_eff512_XLData/eff.log"
BATCH_SIZE=16
SIZE=512
N_EPOCH=40
LR=5e-4
N_FOLDS=5
TTA_FLAG = True
TTA_TIMES = 8
SUBMITNAME = "submission_eff512_XLData"

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PlantDataset(Dataset):
    
    def __init__(self, df, transforms=None):
    
        self.df = df
        self.transforms=transforms
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_src = DIR_INPUT + '/images/' + self.df.loc[idx, 'image_id'] + '.jpg'
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        labels = torch.from_numpy(labels.astype(np.uint8))
        labels = labels.unsqueeze(-1)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, labels


class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load(PRETRAINED_FILE))
        in_features = self.resnet.fc.in_features
        self.logit = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        x = F.dropout(x, 0.25, self.training)

        x = self.logit(x)

        return x

class Efficientb5(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self,x):
        x = self.model(x)
        return x

class DenseCrossEntropy(nn.Module):

    def __init__(self):
        super(DenseCrossEntropy, self).__init__()
        
        
    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        logprobs = F.log_softmax(logits, dim=-1)
        
        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()

transforms_train = A.Compose([
    A.RandomResizedCrop(height=SIZE, width=SIZE, p=1.0),
    A.Flip(),
    A.ShiftScaleRotate(rotate_limit=10, p=0.8),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_valid = A.Compose([
    A.Resize(height=SIZE, width=SIZE, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])
submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')
submission_df.iloc[:, 1:] = 0
if (TTA_FLAG==True):
    dataset_test = PlantDataset(df=submission_df, transforms=transforms_train)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)
else:
    dataset_test = PlantDataset(df=submission_df, transforms=transforms_valid)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)

labeled_df = pd.read_csv(DIR_INPUT + '/train_XL.csv')##############这是输入
train_df = labeled_df
train_labels = train_df.iloc[:, 1:].values
train_y = train_labels[:, 2] + train_labels[:, 3] * 2 + train_labels[:, 1] * 3
dataset_train = PlantDataset(df=train_df, transforms=transforms_train)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)

valid_df = labeled_df.loc[:220]
valid_df.index=pd.Series(range(221))
valid_labels = valid_df.iloc[:, 1:].values
dataset_valid = PlantDataset(df=valid_df, transforms=transforms_valid)
dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)

#model = Efficientb5(num_classes=4)
#model.to(device)

criterion = DenseCrossEntropy()
#optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(model.parameters(),lr=LR)

logf=None

def train_fold(i_fold, model, criterion, optimizer, dataloader_train, dataloader_valid,scheduler):
    print("start_training\n")
    bestscore=0
    bestpred=None
    for epoch in range(N_EPOCH):
        model.train()
        tr_loss = 0
        tr_preds = None
        tr_labels = None
        stepscnt=0
        for step, batch in enumerate(dataloader_train):
            images = batch[0]
            labels = batch[1]
            if tr_labels is None:
                tr_labels = labels.clone().squeeze(-1)
            else:
                tr_labels = torch.cat((tr_labels, labels.squeeze(-1)), dim=0)  

            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            outputs = model(images)
            preds = torch.softmax(outputs, dim=1).data.cpu()
            if tr_preds is None:
                tr_preds = preds
            else:
                tr_preds = torch.cat((tr_preds, preds), dim=0) 

            
            loss = criterion(outputs, labels.squeeze(-1))                
            loss.backward()
            print("train step : {},loss : {}".format(step,loss))
            tr_loss += loss.item()
            stepscnt += 1
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        train_score = roc_auc_score(tr_labels, tr_preds, average='macro')
        print("train EPOCH : {},loss : {}".format(epoch,tr_loss/stepscnt))
        logf.write("train EPOCH : {},loss : {}\n".format(epoch,tr_loss/stepscnt))
        print("train EPOCH : {},train roc_auc: {}".format(epoch,roc_auc_score(tr_labels, tr_preds, average='macro')))
        logf.write("train EPOCH : {},train roc_auc: {}\n".format(epoch,roc_auc_score(tr_labels, tr_preds, average='macro')))
        val_preds = None
        val_labels = None
        
        for step, batch in enumerate(dataloader_valid):
            images = batch[0]
            labels = batch[1]

            if val_labels is None:
                val_labels = labels.clone().squeeze(-1)
            else:
                val_labels = torch.cat((val_labels, labels.squeeze(-1)), dim=0)
            
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)
            with torch.no_grad():
                outputs = model(images)
                preds = torch.softmax(outputs, dim=1).data.cpu()
                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)
        
        val_score=roc_auc_score(val_labels, val_preds, average='macro')
        score = val_score
        if (train_score<val_score):
            score = train_score
        if (score>bestscore):
            bestscore=score
            torch.save(model.state_dict(), '%s/effnet_%03d.pth' % (MODELPATH, i_fold + 1))
            print("score improved")
            logf.write("score improved\n")
            bestpred=val_preds

        print("train EPOCH : {},roc_auc: {}".format(epoch,val_score))
        logf.write("train EPOCH : {},roc_auc: {}\n".format(epoch,val_score))
    
    return bestpred

def train():
    global logf
    logf=open(LOGFILE,"w")
    train_results = []
    submissions = None
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    oof_preds = np.zeros((train_df.shape[0], 4))

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))
        logf.write("Fold {}/{}".format(i_fold + 1, N_FOLDS))
        valid = train_df.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)

        train = train_df.iloc[train_idx]
        train.reset_index(drop=True, inplace=True)    

        dataset_train = PlantDataset(df=train, transforms=transforms_train)
        dataset_valid = PlantDataset(df=valid, transforms=transforms_valid)

        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=8, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)

        model = nn.DataParallel(Efficientb5(num_classes=4))
        model.cuda()   
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20,30],gamma = 0.3)

        val_preds = train_fold(i_fold, model, criterion, optimizer, dataloader_train, dataloader_valid,scheduler)
        oof_preds[valid_idx, :] = val_preds.numpy()
        model.load_state_dict(torch.load('%s/effnet_%03d.pth' % (MODELPATH, i_fold + 1)))
    
        model.eval()
        test_preds=eval(i_fold,model)
        if submissions is None:
            submissions = test_preds / N_FOLDS
        else:
            submissions += test_preds / N_FOLDS
    print("5-Folds CV score: {:.4f}".format(roc_auc_score(train_labels, oof_preds, average='macro')))
    logf.write("5-Folds CV score: {:.4f}\n".format(roc_auc_score(train_labels, oof_preds, average='macro')))
    logf.close()
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(submissions, dim=1)
    submission_df.to_csv("{}.csv".format(SUBMITNAME), index=False)

def eval(i_fold,model):
    test_preds = None

    for step, batch in enumerate(dataloader_test):
        images = batch[0]
            
        images = images.to(device, dtype=torch.float)
        with torch.no_grad():
            outputs = model(images)
            preds = outputs.data.cpu()
            if test_preds is None:
                test_preds = preds
            else:
                test_preds = torch.cat((test_preds, preds), dim=0)
        #print(step)
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_preds,dim=1)
    submission_df.to_csv('{}_fold{}.csv'.format(SUBMITNAME,i_fold),index=False)
    return test_preds
    
def TTA_eval():
    submissions = None
    model = nn.DataParallel(Efficientb5(num_classes=4))
    model.cuda()
    for i_fold in range(5):
        model.load_state_dict(torch.load('%s/effnet_%03d.pth' % (MODELPATH, i_fold + 1)))
        model.eval()
        for j in range(TTA_TIMES):
            test_preds=eval(i_fold,model)
            if submissions is None:
                submissions = test_preds / (N_FOLDS*TTA_TIMES)
            else:
                submissions += test_preds / (N_FOLDS*TTA_TIMES)
        print("fold{} done".format(i_fold));
    
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(submissions, dim=1)
    submission_df.to_csv("{}_TTA.csv".format(SUBMITNAME), index=False)

if __name__=='__main__':
    train()
    #model.load_state_dict(torch.load("%s/resnet_009.pth" % (MODELPATH)))
    #eval()
    TTA_eval()

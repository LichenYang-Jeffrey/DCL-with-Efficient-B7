import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import ImageStat
import PIL.Image as Image
import random

import os
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

from AAPolicy import ImageNetPolicy
import torchvision.transforms as transforms
from PIL import Image
from models.DCL import *
from config import *
from utils.dataset_DCL import collate_fn4train, collate_fn4val, collate_fn4test, collate_fn4backbone
from models.EffNet import EfficientNet

# args
DIR_INPUT = '/data1/yanglichen'
# DIR_INPUT = '/nvme/yangzeyu/plant-pathology-2020-fgvc7/images_mixup/'
# TRAIN_CSV_PATH = '/nvme/yangzeyu/plant-pathology-2020-fgvc7/train_XL_mixup_plus.csv'
PRETRAINED_FILE = "./resnet18-5c106cde.pth"
MODELPATH = "./models_dcleff512_XLData"
LOGFILE = "dcleff.log"
BATCH_SIZE = 8
SIZE = 512
N_EPOCH = 40
LR = 5e-4
N_FOLDS = 5
TTA_FLAG = True
TTA_TIMES = 8
NUM_CLS = 4
resize_resolution=512
crop_resolution=448
swap_num=[7, 7]
SUBMITNAME = "submission_dcleff512_XLData"

device_ids = [0, 1, 2, 3]
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)


class PlantDataset(Dataset):
    def __init__(self, df, swap_size=(7, 7), num_classes=4, common_aug=None, swap=None, totensor=None, train=False,
                 train_val=False, test=False, transforms=None):
        self.df = df
        self.swap = swap
        self.common_aug = common_aug
        self.train = train
        self.test = test
        self.swap_size = swap_size
        self.totensor = totensor
        self.transforms = transforms
        self.numcls = num_classes

    def __len__(self):
        return self.df.shape[0]

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, idx):
        image_src = DIR_INPUT + '/images/' + self.df.loc[idx, 'image_id'] + '.jpg'
        # image_src = DIR_INPUT + self.df.loc[idx, 'image_id'] + '.jpg'
        image = self.pil_loader(image_src)
        label = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        label = label.astype(np.uint8)

        img_unswap = self.common_aug(image) if not self.common_aug is None else image
        image_unswap_list = self.crop_image(img_unswap, self.swap_size)
        swap_range = self.swap_size[0] * self.swap_size[1]
        swap_law1 = [(i - (swap_range // 2)) / swap_range for i in range(swap_range)]
        if self.train:
            img_swap = self.swap(img_unswap)
            image_swap_list = self.crop_image(img_swap, self.swap_size)
            unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list]
            swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]
            swap_law2 = []
            for swap_im in swap_stats:
                distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
                index = distance.index(min(distance))
                swap_law2.append((index - (swap_range // 2)) / swap_range)
            img_swap = self.totensor(img_swap)
            label_swap = np.append(np.array([0]*self.numcls),label)
            img_unswap = self.totensor(img_unswap)
            return img_unswap, img_swap, label, label_swap, swap_law1, swap_law2, image_src
        else:
            swap_law2 = [(i - (swap_range // 2)) / swap_range for i in range(swap_range)]
            label_swap = label
            img_unswap = self.totensor(img_unswap)
            return img_unswap, label, label_swap, swap_law1, swap_law2, image_src


class Efficientb7(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
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


class DenseCrossEntropy_smooth(nn.Module):
    def __init__(self, num_classes=4, label_smoothing=0.05):
        super(DenseCrossEntropy_smooth, self).__init__()
        self.smoothing = label_smoothing
        self.criterion = DenseCrossEntropy()
        self.num_classes = num_classes

    def forward(self, x, target):
        x = x.float()
        traget = target.float()
        assert x.size(1) == self.num_classes
        target_smooth = ((1 - self.smoothing) * target) + (self.smoothing / self.num_classes)

        return self.criterion(x, target_smooth)


transformers = load_data_transformers(resize_resolution, crop_resolution, swap_num)

#submission_df = pd.read_csv(DIR_INPUT + '/sample_submission.csv')
submission_df = pd.read_csv('./sample_submission.csv')
submission_df.iloc[:, 1:] = 0

dataset_test = PlantDataset(df=submission_df,
                            num_classes=NUM_CLS,
                            common_aug=transformers["None"],
                            swap=transformers["None"],
                            totensor=transformers["test_totensor"],
                            test=True)
dataloader_test = DataLoader(dataset_test,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=8,
                             collate_fn=collate_fn4test,
                             drop_last=False,
                             pin_memory=True)

setattr(dataloader_test, 'total_item_len', len(dataset_test))
setattr(dataloader_test, 'num_cls', NUM_CLS)

labeled_df = pd.read_csv(DIR_INPUT + '/train_XL.csv')
# labeled_df = pd.read_csv(TRAIN_CSV_PATH)
train_df = labeled_df
train_labels = train_df.iloc[:, 1:].values
train_y = train_labels[:, 2] + train_labels[:, 3] * 2 + train_labels[:, 1] * 3
dataset_train = PlantDataset(df=train_df,
                             num_classes=NUM_CLS,
                             common_aug=transformers["common_aug"],
                             swap=transformers["swap"],
                             totensor=transformers["train_totensor"],
                             train=True)
dataloader_train = DataLoader(dataset_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=8,
                              collate_fn=collate_fn4train,
                              drop_last=False,
                              pin_memory=True)
setattr(dataloader_train, 'total_item_len', len(dataset_train))

valid_df = labeled_df.loc[:220]
valid_df.index = pd.Series(range(221))
valid_labels = valid_df.iloc[:, 1:].values
dataset_valid = PlantDataset(df=valid_df,
                             num_classes=NUM_CLS,
                             common_aug=transformers["None"],
                             swap=transformers["None"],
                             totensor=transformers["val_totensor"],
                             train=False,
                             train_val=True)
dataloader_valid = DataLoader(dataset_valid,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=8,
                              collate_fn=collate_fn4test,
                              drop_last=False,
                              pin_memory=True)
setattr(dataloader_valid, 'total_item_len', len(dataset_valid))
setattr(dataloader_valid, 'num_cls', NUM_CLS)


criterion4 = DenseCrossEntropy_smooth(num_classes=4)
criterion8 = DenseCrossEntropy_smooth(num_classes=8)

def train_fold(i_fold, model, optimizer, dataloader_train, dataloader_valid, scheduler, logf):
    print("start_training\n")
    bestscore = 0
    bestpred = None
    train_batch_size = dataloader_train.batch_size
    add_loss = nn.L1Loss()
    for epoch in range(N_EPOCH):
        model.train(True)
        tr_loss = 0
        tr_preds = None
        tr_labels = None
        stepscnt = 0
        for step, batch in enumerate(dataloader_train):
            loss = 0.0
            model.train(True)

            inputs, labels, labels_swap, swap_law, img_names = batch
            labels = torch.from_numpy(np.array(labels))

            if tr_labels is None:
                tr_labels = labels.clone().squeeze(-1)
            else:
                tr_labels = torch.cat((tr_labels, labels.clone().squeeze(-1)), dim=0)

            inputs = Variable(inputs.to(device, dtype=torch.float))
            labels = Variable(labels.to(device, dtype=torch.long))
            labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).to(device, dtype=torch.float))
            swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().to(device, dtype=torch.float))

            optimizer.zero_grad()

            '''
            if inputs.size(0) < 2 * train_batch_size:
                outputs = model(inputs, inputs[0:-1:2])
            else:
                outputs = model(inputs, None)
            '''

            outputs = model(inputs, None)

            ce_loss = criterion4(outputs[0], labels)
            loss += ce_loss

            preds = torch.softmax(outputs[0], dim=1).data.cpu()
            if tr_preds is None:
                tr_preds = preds
            else:
                tr_preds = torch.cat((tr_preds, preds), dim=0)

            alpha_ = 1.0
            beta_ = 1.0
            gamma_ = 1.0

            swap_loss = criterion8(outputs[1], labels_swap) * beta_
            loss += swap_loss
            law_loss = add_loss(outputs[2], swap_law) * gamma_
            loss += law_loss

            tr_loss += loss

            loss.backward()
            torch.cuda.synchronize()
            print("train step : {}, loss : {}".format(step, loss))
            stepscnt += 1
            optimizer.step()
            torch.cuda.synchronize()

        scheduler.step()
        train_score = roc_auc_score(tr_labels, tr_preds, average='macro')
        print("train EPOCH : {},loss : {}".format(epoch, tr_loss / stepscnt))
        logf.write("train EPOCH : {},loss : {}\n".format(epoch, tr_loss / stepscnt))
        print("train EPOCH : {},train roc_auc: {}".format(epoch, roc_auc_score(tr_labels, tr_preds, average='macro')))
        logf.write(
            "train EPOCH : {},train roc_auc: {}\n".format(epoch, roc_auc_score(tr_labels, tr_preds, average='macro')))
        logf.flush()

        model.train(False)
        val_preds = None
        val_labels = None
        with torch.no_grad():
            for step, batch in enumerate(dataloader_valid):
                inputs = batch[0]
                labels = torch.from_numpy(np.array(batch[1]))
                if val_labels is None:
                    val_labels = labels.clone().squeeze(-1)
                else:
                    val_labels = torch.cat((val_labels, labels.clone().squeeze(-1)), dim=0)
                inputs = Variable(inputs.to(device, dtype=torch.float))
                labels = Variable(labels.to(device, dtype=torch.long))
                outputs = model(inputs)
                preds = torch.softmax(outputs[0], dim=1).data.cpu()
                if val_preds is None:
                    val_preds = preds
                else:
                    val_preds = torch.cat((val_preds, preds), dim=0)
        val_score = roc_auc_score(val_labels, val_preds, average='macro')
        score = val_score
        if (train_score < val_score):
            score = train_score
        if (score > bestscore):
            bestscore = score
            torch.save(model.state_dict(), '%s/dcleffnet_%03d.pth' % (MODELPATH, i_fold + 1))
            print("score improved")
            logf.write("score improved\n")
            logf.flush()
            bestpred = val_preds

        print("train EPOCH : {},roc_auc: {}".format(epoch, val_score))
        logf.write("train EPOCH : {},roc_auc: {}\n".format(epoch, val_score))
        logf.flush()

    return bestpred


def train(logf):
    train_results = []
    submissions = None
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    oof_preds = np.zeros((train_df.shape[0], 4))

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))
        logf.write("Fold {}/{}".format(i_fold + 1, N_FOLDS))
        logf.flush()
        valid = train_df.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)

        train = train_df.iloc[train_idx]
        train.reset_index(drop=True, inplace=True)

        dataset_train = PlantDataset(df=train,
                                     num_classes=NUM_CLS,
                                     common_aug=transformers["common_aug"],
                                     swap=transformers["swap"],
                                     totensor=transformers["train_totensor"],
                                     train=True)
        dataset_valid = PlantDataset(df=valid,
                                     num_classes=NUM_CLS,
                                     common_aug=transformers["None"],
                                     swap=transformers["None"],
                                     totensor=transformers["val_totensor"],
                                     train=False,
                                     train_val=True)
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=8,
                                      collate_fn=collate_fn4train,
                                      drop_last=False,
                                      pin_memory=True)
        setattr(dataloader_train, 'total_item_len', len(dataset_train))
        dataloader_valid = DataLoader(dataset_valid,
                                      batch_size=BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=8,
                                      collate_fn=collate_fn4test,
                                      drop_last=False,
                                      pin_memory=True)
        setattr(dataloader_valid, 'total_item_len', len(dataset_valid))
        setattr(dataloader_valid, 'num_cls', NUM_CLS)

        backbone = Efficientb7(num_classes=NUM_CLS)
        #backbone.load_state_dict(torch.load('./models_eff512_XLData/effnet_%03d.pth' % (i_fold + 1)))
        model = nn.DataParallel(MainModel(backbone=backbone, numcls=NUM_CLS), device_ids=device_ids)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.3)

        val_preds = train_fold(i_fold, model, optimizer, dataloader_train, dataloader_valid, scheduler,logf=logf)
        oof_preds[valid_idx, :] = val_preds.numpy()

        model.eval()
        test_preds = eval(i_fold, model)
        if submissions is None:
            submissions = test_preds / N_FOLDS
        else:
            submissions += test_preds / N_FOLDS
    print("5-Folds CV score: {:.4f}".format(roc_auc_score(train_labels, oof_preds, average='macro')))
    logf.write("5-Folds CV score: {:.4f}\n".format(roc_auc_score(train_labels, oof_preds, average='macro')))
    logf.close()
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(submissions, dim=1)
    submission_df.to_csv("{}.csv".format(SUBMITNAME), index=False)


def eval(i_fold, model):
    test_preds = None

    for step, batch in enumerate(dataloader_test):
        images = batch[0]

        images = images.to(device, dtype=torch.float)
        with torch.no_grad():
            outputs = model(images)
            preds = outputs[0] + outputs[1][:,0:NUM_CLS] + outputs[1][:,NUM_CLS:2*NUM_CLS]
            preds = preds.data.cpu()
            if test_preds is None:
                test_preds = preds
            else:
                test_preds = torch.cat((test_preds, preds), dim=0)
        # print(step)
    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(test_preds, dim=1)
    submission_df.to_csv('{}_fold{}.csv'.format(SUBMITNAME, i_fold), index=False)
    return test_preds


def TTA_eval():
    submissions = None
    backbone = Efficientb7(num_classes=NUM_CLS)
    model = nn.DataParallel(MainModel(backbone=backbone, numcls=NUM_CLS), device_ids=device_ids)
    model.to(device)
    for i_fold in range(5):
        #model.load_state_dict(torch.load('%s/effnet_%03d.pth' % (MODELPATH, i_fold + 1)))
        model.load_state_dict(torch.load('%s/dcleffnet_%03d.pth' % (MODELPATH, i_fold + 1)))
        model.eval()
        for j in range(TTA_TIMES):
            test_preds = eval(i_fold, model)
            if submissions is None:
                submissions = test_preds / (N_FOLDS * TTA_TIMES)
            else:
                submissions += test_preds / (N_FOLDS * TTA_TIMES)
        print("fold{} done".format(i_fold));

    submission_df[['healthy', 'multiple_diseases', 'rust', 'scab']] = torch.softmax(submissions, dim=1)
    submission_df.to_csv("{}_TTA.csv".format(SUBMITNAME), index=False)


if __name__ == '__main__':
    logf = open(LOGFILE, "w")
    train(logf)
    TTA_eval()
    logf.close()

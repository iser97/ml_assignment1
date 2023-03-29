import optuna
from optuna.trial import TrialState
from utils import *
from tqdm import tqdm
import argparse
import logging
import os
import sys
from sklearn import model_selection
from sklearn import metrics
import torch.nn as nn
import torchaudio
import torch
import librosa
from modules import *
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed
from collections.abc import Mapping
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
set_seed(100)
from transformers import ASTForAudioClassification
from collections.abc import Mapping
def _prepare_input(data):
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.squeeze().to(device)
    return data

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

class MusicnnDataset(Dataset):
    def __init__(self, batch_size=10, istrain=True, data_type='mfcc') -> None:
        super().__init__()
        self.data_type = data_type
        self.istrain = istrain
        self.train_info, self.valid_info, self.train_classes, self.valid_classes = model_selection.train_test_split(
            train_info, train_classes, train_size=0.8, test_size=0.2, random_state=4487
        )
        # self.valid_info = train_info
        # self.valid_classes = train_classes
        if data_type=='raw':
            self.input_length = 16000*3
        elif data_type=='mfcc':
            self.input_length = int(16000*3/256)
    def __len__(self):
        if self.istrain:
            return len(self.train_info)
        else:
            return len(self.valid_info)
    
    @staticmethod
    def mp3_expand(raw, input_length):
        length = len(raw)
        factor = input_length // length + 1
        raw = [raw for _ in range(factor)]
        raw = np.stack(raw, axis=0)
        raw = raw.flatten()
        return raw
    
    @staticmethod
    def mfcc_expand(mfcc, input_length):
        length = mfcc.shape[2]
        factor = input_length // length + 1
        start = mfcc
        for i in range(factor):
            start = np.concatenate([start, mfcc], axis=2)
        start = start[:, :, :input_length]
        return start

    def __getitem__(self, index):
        if self.istrain:
            file_id = self.train_info[index]['id']
            label = self.train_classes[index]
        else:
            file_id = self.valid_info[index]['id']
            label = self.valid_classes[index]
        file_root = os.path.join(music_raw_root, '{}.npy'.format(file_id))
        # spec_file_root = os.path.join(music_raw_root, '{}_mel.npy'.format(file_id))
        mfcc_file_root = os.path.join(music_raw_root, '{}_mfcc.npy'.format(file_id))
        
        if self.data_type=='raw':
            npy = np.load(file_root)
            if len(npy) < self.input_length:
                npy = self.mp3_expand(npy, self.input_length)
            random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
            data = np.array(npy[random_idx:random_idx+self.input_length], dtype=np.float32)
        
        elif self.data_type=='mfcc':
            mfcc_stack = np.load(mfcc_file_root)
            if mfcc_stack.shape[2]<self.input_length:
                mfcc_stack = self.mfcc_expand(mfcc_stack, self.input_length)
            random_idx = int(np.floor(np.random.random(1) * (mfcc_stack.shape[2]-self.input_length)))
            data = np.array(mfcc_stack[:, :, random_idx:random_idx+self.input_length], dtype=np.float32)
        
        return data, np.float32(label)

class PredTestData(Dataset):
    def __init__(self, batch_size, data_type='mfcc') -> None:
        super().__init__()
        self.data_type = data_type
        self.testinfo = test_info
        if data_type=='raw':
            self.input_length = 16000*3
        elif data_type=='mfcc':
            self.input_length = int(16000*3/256)
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.testinfo)

    def __getitem__(self, index):
        fileid = self.testinfo[index]['id']
        file_root = os.path.join(music_raw_root, '{}.npy'.format(fileid))
        mfcc_file_root = os.path.join(music_raw_root, '{}_mfcc.npy'.format(fileid))
        if self.data_type=='raw':
            raw = np.load(file_root)
            length = len(raw)
            if length<self.input_length:
                raw = MusicnnDataset.mp3_expand(raw, self.input_length)
            hop = (length - self.input_length) // self.batch_size
            x = torch.zeros(self.batch_size, self.input_length)
            for i in range(self.batch_size):
                x[i] = torch.Tensor(np.float32(raw[i*hop:i*hop+self.input_length])).unsqueeze(0)
        elif self.data_type=='mfcc':
            mfcc_stack = np.load(mfcc_file_root)
            if mfcc_stack.shape[2]<self.input_length:
                mfcc_stack = MusicnnDataset.mfcc_expand(mfcc_stack, self.input_length)
            channel, mfcc_filters, length = mfcc_stack.shape
            hop = (length - self.input_length) // self.batch_size
            x = torch.zeros(self.batch_size, channel, mfcc_filters, self.input_length)
            for i in range(self.batch_size):
                x[i] = torch.Tensor(mfcc_stack[:, :, i*hop:i*hop+self.input_length])
        return x


class Musicnn(nn.Module):
    '''
    Pons et al. 2017
    End-to-end learning for music audio tagging at scale.
    This is the updated implementation of the original paper. Referred to the Musicnn code.
    https://github.com/jordipons/musicnn
    '''
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=19,
                dataset='mtat'):
        super(Musicnn, self).__init__()
        self.n_class = n_class
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(3)

        # Pons front-end
        m1 = Conv_V(3, 204, (int(0.7*96), 7))
        m2 = Conv_V(3, 204, (int(0.4*96), 7))
        m3 = Conv_H(3, 51, 129)
        m4 = Conv_H(3, 51, 65)
        m5 = Conv_H(3, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # Pons back-end
        backend_channel= 512 if dataset=='msd' else 64
        self.layer1 = Conv_1d(561, backend_channel, 7, 1, 1)
        self.layer2 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)
        self.layer3 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)

        # Dense
        dense_channel = 500 if dataset=='msd' else 200
        self.dense1 = nn.Linear((561+(backend_channel*3))*2, dense_channel)
        self.bn = nn.BatchNorm1d(dense_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_channel, n_class)

    def forward(self, x):
        # Spectrogram
        # x = self.spec(x)
        # x = self.to_db(x)
        # x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # Pons front-end
        out = []
        for layer in self.layers:
            out.append(layer(x))
        out = torch.cat(out, dim=1)

        # Pons back-end
        length = out.size(2)
        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)

        out = torch.cat([mp, avgp], dim=1)
        out = out.squeeze(2)

        out = self.relu(self.bn(self.dense1(out)))
        out = self.dropout(out)
        out = self.dense2(out)
        out = nn.Sigmoid()(out)

        return out

def pred_test(args, model, loader_test):
    model.eval()
    est_lst = []
    for data in loader_test:
        data = data.squeeze()
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
        pred = pred.detach().cpu().numpy()
        estimated = pred.mean(axis=0)
        est_lst.append(estimated)
    est_lst = np.array(est_lst)
    return est_lst

def main(args):
    best_metric = 0
    dataset_train = MusicnnDataset(batch_size=args.batch_size, istrain=True)
    dataset_eval = MusicnnDataset(batch_size=args.batch_size, istrain=False)
    dataset_test = PredTestData(batch_size=args.batch_size)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size)
    loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)
    loss_fct = nn.BCELoss()
    model = Musicnn(n_class=19)
    model = model.to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in tqdm(range(args.max_epoch)):
        model.train()
        for data, label in loader_train:
            # label = torch.Tensor(label).long()
            model.zero_grad()
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            loss = loss_fct(pred, label)
            loss.backward()
            optimizer.step()
        model.eval()
        label_stack = []
        losses = []
        pred_stack = torch.randn(size=(1, model.n_class)).to(device)
        for data, label in loader_eval:
            # label = torch.Tensor(label).long()
            data = data.float()
            data = data.to(device)
            label = label.to(device)
            with torch.no_grad():
                pred = model(data)
                pred_stack = torch.cat((pred_stack, pred), dim=0)
                loss = loss_fct(pred, label)
                losses.append(loss.item())
            label_stack += label.cpu().numpy().tolist()
        cur_loss = np.array(losses).mean(0)
        cur_metric = 1 - cur_loss
        pred_stack = pred_stack[1:]
        pred_stack = pred_stack.detach().cpu().numpy()
        label_stack = np.array(label_stack)
        # roc_aucs = plot_roc(tagnames, label_stack, pred_stack)
        roc_aucs = metrics.roc_auc_score(label_stack, pred_stack, average='macro')
        pr_aucs = metrics.average_precision_score(label_stack, pred_stack, average='macro')
        # print('cur_metric: {%.4f}   roc_auc: {%.4f},     pr_auc: {%.4f}'.format(cur_metric, roc_aucs, pr_aucs))
        if epoch % 10 == 0:
            print('cur_metric: {:.4} ---- roc_aucs: {:.4} ---- pr_auc: {:.4}'.format(cur_metric, roc_aucs, pr_aucs))
        if roc_aucs > best_metric:
            best_metric = roc_aucs
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            pred_score = pred_test(args, model, loader_test)
    return best_metric, pred_score

def objective(trial):
    args.lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    args.batch_size = trial.suggest_int('batch size', 32, 64)

    args.save_dir = os.path.join(experiment_dir, "trial_{}".format(trial.number))
    os.makedirs(args.save_dir, exist_ok=True)
    
    for key, value in trial.params.items():
        print("  {}: {} \n".format(key, value))
    best_acc, pred_score = main(args)
    
    with open(os.path.join(args.save_dir, "best_acc.txt"), mode='w', encoding='utf-8') as w:
        for key, value in trial.params.items():
            w.writelines("    {}: {} \n".format(key, value))
        w.writelines(str(best_acc))
        
    if best_acc > args.best_acc:
        args.best_acc = best_acc
        args.best_trial = trial.number
        write_csv_kaggle_tags(os.path.join(experiment_dir, '{}.csv'.format("my_submission")), tagnames, pred_score)
    return best_acc


def do_trial(args):
    study = optuna.create_study(directions=['maximize'])
    study.optimize(objective, n_trials=args.n_trials)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
if __name__ == '__main__':
    music_raw_root = './musicmp3'
    train_tags  = load_pickle('musicdata/train_tags.pickle3')
    train_mfccs = load_pickle('musicdata/train_mfccs.pickle3')
    train_mels  = load_pickle('musicdata/train_mels.pickle3')
    train_info  = load_pickle('musicdata/train_info.pickle3')

    test_mfccs = load_pickle('musicdata/test_mfccs.pickle3')
    test_mels  = load_pickle('musicdata/test_mels.pickle3')
    test_info  = load_pickle('musicdata/test_info.pickle3')
    
    tagnames, tagnames_counts = np.unique(np.concatenate(train_tags), return_counts=True)
    train_classes = tags2class(train_tags, tagnames)


    parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                        help='how many epochs')
    parser.add_argument('--experiment_name', type=str, default='triplet_training',
                        help='experiment name')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='number of trials')
    parser.add_argument('--best_acc', type=float, default=0,
                        help='number of trials')
    parser.add_argument('--best_trial', type=int, default=0)
    args = parser.parse_args()
    
    experiment_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=False)
    do_trial(args)
    print(args)
    
import os
import sys
import optuna
from optuna.trial import TrialState
from utils import *
from tqdm import tqdm
import argparse
import logging

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

from transformers import ASTPreTrainedModel, ASTModel, AutoConfig, AutoFeatureExtractor, ASTForAudioClassification

class DenseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, 19)
    
    def forward(self, hidden_state):
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state


class ASTagModel(ASTPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.audio_spectrogram_transformer = ASTModel(config)
        self.linear = DenseLayer(config)
        self.args = kwargs['model_args']
        self.n_class = 19
    def forward(self, input_values):
        outputs = self.audio_spectrogram_transformer(input_values)
        hidden_states = outputs.last_hidden_state
        if self.args.pooler == 1:
            pool_output = torch.mean(hidden_states, dim=1)
        elif self.args.pooler == 0:
            pool_output = outputs.pooler_output
        logits = self.linear(pool_output)
        return nn.Sigmoid()(logits)

class ASTDataset(Dataset):
    def __init__(self, mode='train', cross_eval=True):
        super().__init__()
        self.mode = mode
        if not cross_eval:
            self.train_info = train_info
            self.valid_info = train_info
            self.train_classes = train_classes
            self.valid_classes = train_classes
        else:
            self.train_info, self.valid_info, self.train_classes, self.valid_classes = model_selection.train_test_split(
                train_info, train_classes, train_size=0.8, test_size=0.2, random_state=4487
            )
        self.test_info = test_info
    
    def __len__(self):
        if self.mode=='train':
            return(len(self.train_info))
        elif self.mode=='eval':
            return(len(self.valid_info))
        elif self.mode=='test':
            return(len(self.test_info))
    
    def __getitem__(self, index):
        if self.mode=='test':
            file_id = self.test_info[index]['id']
            ast_root = os.path.join(music_raw_root, '{}_ast.npy'.format(file_id))
            ast_feature = np.load(ast_root)
            return ast_feature
        else:
            if self.mode=='train':
                file_id = self.train_info[index]['id']
                label = self.train_classes[index]
            elif self.mode=='eval':
                file_id = self.valid_info[index]['id']
                label = self.valid_classes[index]
            file_root = os.path.join(music_raw_root, '{}.npy'.format(file_id))
            ast_root = os.path.join(music_raw_root, '{}_ast.npy'.format(file_id))
            ast_feature = np.load(ast_root)
            return ast_feature.squeeze(), np.float32(label)

def pred_test(args, model, loader_test):
    model.eval()
    result = np.random.randn(1, 19)
    est_lst = []
    for data in loader_test:
        data = data.squeeze()
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
        pred = pred.detach().cpu().numpy()
        result = np.concatenate((result, pred), axis=0)
    result = result[1:]
    return result


def main(args):
    best_metric = 0
    
    # dataset_train = MusicnnDataset(batch_size=args.batch_size, istrain=True)
    # dataset_eval = MusicnnDataset(batch_size=args.batch_size, istrain=False)
    # dataset_test = PredTestData(batch_size=args.batch_size)
    # loader_train = DataLoader(dataset_train, batch_size=args.batch_size)
    # loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, drop_last=False)

    dataset_train = ASTDataset(mode='train', cross_eval=True)
    dataset_eval = ASTDataset(mode='eval', cross_eval=True)
    dataset_test = ASTDataset(mode='test')
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size)
    loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, drop_last=False)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    loss_fct = nn.BCELoss()
    # model = Musicnn(n_class=19)
    # model = model.to(device)
    config = AutoConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")  
    model = ASTagModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        config=config,
        model_args=args,
    )
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
        if epoch % 2 == 0:
            print('cur_metric: {:.4} ---- roc_aucs: {:.4} ---- pr_auc: {:.4}'.format(cur_metric, roc_aucs, pr_aucs))
        if roc_aucs > best_metric:
            best_metric = roc_aucs
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            pred_score = pred_test(args, model, loader_test)
    return best_metric, pred_score

def objective(trial):
    args.lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    args.batch_size = trial.suggest_int('batch size', 3, 7)
    args.save_dir = os.path.join(experiment_dir, "trial_{}".format(trial.number))
    args.pooler = trial.suggest_int('pooler', 0, 1)
    
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
    parser.add_argument('--max_epoch', type=int, default=20, metavar='N',
                        help='how many epochs')
    parser.add_argument('--experiment_name', type=str, default='triplet_training',
                        help='experiment name')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='number of trials')
    parser.add_argument('--best_acc', type=float, default=0,
                        help='number of trials')
    parser.add_argument('--best_trial', type=int, default=0)
    parser.add_argument('--pooler', type=int, default=0)
    args = parser.parse_args()
    
    experiment_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=False)
    do_trial(args)
    print(args)
    
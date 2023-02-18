import optuna
from optuna.trial import TrialState
from transformers import AutoConfig
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, XLNetModel
from transformers import XLNetPreTrainedModel
from util import read_text_data, write_csv_kaggle_sub
from tqdm import tqdm
import argparse
import logging
import os
import sys
from sklearn import model_selection
from util import Lstm_Bi

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

class Vector_Dot(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        ) 
        
    def forward(self, x, model_args=None):
        x = self.dense(x)
        return x

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

class XLNetClassModel(XLNetPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs['model_args']
        self.transformer = XLNetModel(config)
        self.lstm = Lstm_Bi(config.hidden_size, config.hidden_size, 1, self.model_args.batch_size, biFlag=False)
        self.linear_proj = Vector_Dot(config.hidden_size, 4)
        self.linear_review = Vector_Dot(config.hidden_size, 4)
        self.pooler = Pooler(pooler_type="avg")
        self.init_weights()
    def forward(self,
                input_ids=None,
                id=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                labels_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                returnd_dict=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        # last_hidden = self.lstm(last_hidden)
        outputs = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        pred_review = self.linear_review(outputs)
        outputs = self.linear_proj(outputs)
        return outputs, pred_review

class BertDataset(Dataset):
    def __init__(self, tokenizer, istrain=True) -> None:
        super().__init__()
        self.train_features = []
        self.test_features = []
        self.id_lst = list(set(trainrevid))
        self.istrain=istrain
        train_txt_split, test_txt_split, self.Y_train_split, self.Y_test_split, self.id_train, self.id_test = model_selection.train_test_split(
            traintxt, 
            trainY, 
            trainrevid,
            train_size=0.7, 
            test_size=0.3, 
            random_state=4487)
        if istrain:
            for index, text in enumerate(train_txt_split):
                text = self.id_train[index] + ' ' +text
                feature = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
                feature = {k:feature[k] for k in feature}
                feature['labels'] = self.Y_train_split[index]
                self.train_features.append(feature)
        else:
            for index, text in enumerate(test_txt_split):
                text = self.id_test[index] + ' ' + text
                feature = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
                feature = {k:feature[k] for k in feature}
                feature['labels'] = self.Y_test_split[index]
                self.test_features.append(feature)
        
    def __len__(self):
        if self.istrain:
            return len(self.train_features)
        else:
            return len(self.test_features)
    
    def __getitem__(self, index):
        
        if self.istrain:
            feature = self.train_features[index]
            feature['id'] = self.id_lst.index(self.id_train[index])
        else:
            feature = self.test_features[index]
        return feature
    
class BertDatasetKaggle(Dataset):
    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.features = []
        for index, text in enumerate(testtxt):
            review_id = testrevid[index]
            text = review_id + ' ' + text
            feature = tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
            self.features.append(feature)
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

from collections.abc import Mapping
def _prepare_input(data):
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.squeeze().to(device)
    return data
    
def pred_test(args, model, save_name):
    pred_class = []
    for data in loader_kaggle:
        data = _prepare_input(data)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            pred, _ = model(**data)
            pred = torch.argmax(pred, dim=-1)
            pred = pred.cpu().numpy().tolist()
            pred_class = pred_class + pred
    write_csv_kaggle_sub(os.path.join(args.save_dir, '{}.csv'.format(save_name)), pred_class)
    
def main(args):
    best_acc = 0
    config = AutoConfig.from_pretrained("xlnet-base-cased")
    xlnetmodel = XLNetClassModel.from_pretrained(
        'xlnet-base-cased',
        config=config,
        model_args=args,
    ) 
    optimizer = torch.optim.Adam(xlnetmodel.parameters(), lr=args.lr, weight_decay=0.0005)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_test = DataLoader(dataset_test, batch_size=20, shuffle=False, drop_last=False)
    
    for epoch in tqdm(range(args.max_epoch)):
        for data in loader_train:
            data = _prepare_input(data)
            xlnetmodel = xlnetmodel.to(device)
            xlnetmodel.train()
            xlnetmodel.zero_grad()
            pred, review = xlnetmodel(**data)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(pred, data['labels'])
            # loss_id = loss_fct(review, data['id'])
            # loss = loss+loss_id
            loss.backward()
            optimizer.step()
        
        correct = 0
        size = 0
        for data in loader_test:
            data = _prepare_input(data)
            label = data['labels']
            size += len(label)
            xlnetmodel = xlnetmodel.to(device)
            xlnetmodel.eval()
            with torch.no_grad():
                pred, _ = xlnetmodel(**data)
                pred = torch.argmax(pred, dim=-1)
                correct += pred.eq(label).cpu().sum()
        now_acc = correct/size
        print('val acc: ', correct/size)
        if now_acc>=best_acc:
            best_acc = now_acc
            pred_test(args, xlnetmodel, 'xlnet_pred')
    return best_acc

def objective(trial):
    args.lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    args.batch_size = trial.suggest_int('batch size', 10, 16)
    args.max_epoch = trial.suggest_int('max epoch', 10, 30)

    args.save_dir = os.path.join(experiment_dir, "trial_{}".format(trial.number))
    os.makedirs(args.save_dir, exist_ok=True)
    
    for key, value in trial.params.items():
        logger.info("  {}: {} \n".format(key, value))
        
    best_acc = main(args)
    with open(os.path.join(args.save_dir, "best_acc.txt"), mode='w', encoding='utf-8') as w:
        for key, value in trial.params.items():
            w.writelines("    {}: {} \n".format(key, value))
        w.writelines(str(best_acc))

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
    (traintxt, trainrevid, trainY) = read_text_data("movierating_train.txt")
    (testtxt, testrevid, _)        = read_text_data("movierating_test.txt")

    xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    dataset_train = BertDataset(xlnet_tokenizer, istrain=True)
    dataset_test = BertDataset(xlnet_tokenizer, istrain=False)
    dataset_kaggle = BertDatasetKaggle(xlnet_tokenizer)
    loader_train = DataLoader(dataset_train, batch_size=13, shuffle=True, drop_last=True)
    loader_test = DataLoader(dataset_test, batch_size=10, shuffle=False, drop_last=False)
    loader_kaggle = DataLoader(dataset_kaggle, batch_size=20, drop_last=False)

    parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--max_epoch', type=int, default=50, metavar='N',
                        help='how many epochs')
    parser.add_argument('--experiment_name', type=str, default='triplet_training',
                        help='experiment name')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='number of trials')
    args = parser.parse_args()
    
    experiment_dir = os.path.join("experiments", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=False)
    do_trial(args)
    
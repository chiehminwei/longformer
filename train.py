import os
import shutil
from tensorboardX import SummaryWriter
import numpy as np
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import MafiascumDataset
from transformers import LongformerModel, LongformerTokenizer, LongformerConfig, AdamW, get_linear_schedule_with_warmup
from sklearn import metrics


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Longformer""")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument("--num_warmup_steps", type=int, default=500)
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--train_set", type=str, default="data/train.pkl")
    parser.add_argument("--test_set", type=str, default="data/test.pkl")
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard/longformer")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    return args


def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def pad_to_window_size(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = 2 * one_sided_window_size
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask


class LongformerForBinaryClassification(nn.Module):
    def __init__(self, config):
        super(LongformerForBinaryClassification, self).__init__()
        self.config = config
        self.tokenizer = LongformerTokenizer.from_pretrained('longformer-base-4096')
        self.longformer = LongformerModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, self.config.attention_window[0], self.tokenizer.pad_token_id)

        pooled = self.longformer(input_ids, attention_mask)[1]
        logits = self.classifier(pooled)
        return logits


def train(opt):
    # Set random seed for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
    else:
        torch.manual_seed(opt.seed)

    if opt.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            opt.gradient_accumulation_steps))
    opt.batch_size = opt.batch_size // opt.gradient_accumulation_steps

    # Logging
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))    
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    # Data Loading
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}

    train_data_path = opt.train_set
    training_set = MafiascumDataset(train_data_path)
    training_generator = DataLoader(training_set, **training_params)

    test_data_path = opt.test_set
    test_set = MafiascumDataset(test_data_path)
    test_generator = DataLoader(test_set, **test_params)

    # Model
    config = LongformerConfig.from_pretrained('longformer-base-4096')
    config.attention_mode = 'sliding_chunks'
    model = LongformerForBinaryClassification(config)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.BCEWithLogitsLoss()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, weight_decay=0.01, correct_bias=False)
    num_train_optimization_steps = int(
        len(training_generator) / opt.batch_size / opt.gradient_accumulation_steps) * opt.num_epoches
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                    num_training_steps = num_train_optimization_steps,
                    num_warmup_steps=opt.num_warmup_steps)

    # Training
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epoches):
        for iteration, (input_ids, attention_mask, labels) in enumerate(training_generator):
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda() 
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            if opt.gradient_accumulation_steps > 1:
                loss = loss / opt.gradient_accumulation_steps
            loss.backward()
            if (iteration + 1) % opt.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                training_metrics = get_evaluation(label.cpu().numpy(), logits.cpu().detach().numpy(), list_metrics=["accuracy"])            
                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    iteration + 1,
                    num_iter_per_epoch,
                    optimizer.param_groups[0]['lr'],
                    loss, training_metrics["accuracy"]))
                writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iteration)
                writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iteration)
        
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for (input_ids, attention_mask, labels) in test_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                    labels = labels.cuda()
                with torch.no_grad():
                    logits = model(input_ids, attention_mask=attention_mask)
                te_loss = criterion(logits, labels)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(labels.clone().cpu())
                te_pred_ls.append(logits.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1, opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            writer.add_scalar('Test/Loss', te_loss, epoch)
            writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            # Update if new best loss achieved
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + os.sep + "longformer")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break

    writer.close()
    output_file.close()

if __name__ == "__main__":
    opt = get_args()
    train(opt)


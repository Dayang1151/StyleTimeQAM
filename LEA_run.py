import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

def train(params, model, sentence_all, user_all, timestamp_all,label_all,match_all, optimizer, criterion,
          max_gradient_norm,first_weight):
    model.train()
    device = model.device
    N = len(label_all) // params.batch_size
    running_loss = 0.0
    match_all_prob = []
    match_all_labels = []
    label_all_prob = []
    label_all_labels = []
    time_start = time.time()
    for i in range(N):
        input_sen = sentence_all[i * params.batch_size:(i + 1) * params.batch_size + 200].to(device)
        input_user = user_all[i * params.batch_size:(i + 1) * params.batch_size + 100].to(device)
        label = label_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
        match = match_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
        input_timestamp = timestamp_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)

        optimizer.zero_grad()
        label_logits, label_probs, match_logits,match_probs = model(input_sen, input_user, input_timestamp,params.t_p)
        ###################################################
        loss_label = criterion(label_logits, label)
        loss_match = criterion(match_logits, match.reshape(-1))
        loss = (loss_label * first_weight) + loss_match
        loss.backward()
        running_loss += loss.item()
        ###################################################
        # loss_match = criterion(match_logits, match.reshape(-1))
        # loss_match.backward()
        # running_loss += loss.item()
        ###################################################
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        match_all_prob.extend(match_probs[:, 1].cpu().detach().numpy())
        for i in range(len(match)):
            match_all_labels.extend(match[i].cpu())
        label_all_prob.extend(label_probs[:, 1].cpu().detach().numpy())
        label_all_labels.extend(label.cpu())
    epoch_loss = running_loss / len(sentence_all)
    total_time = time.time() - time_start
    return total_time,epoch_loss,roc_auc_score(label_all_labels,label_all_prob),roc_auc_score(match_all_labels, match_all_prob)


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def valid(params, model, sentence_all, user_all, timestamp_all,label_all,match_all, criterion,first_weight):
    model.eval()
    device = model.device
    N = len(label_all) // params.batch_size
    running_loss = 0.0
    match_all_prob = []
    match_all_labels = []
    label_all_prob = []
    label_all_labels = []
    time_start = time.time()
    for i in range(N):
        input_sen = sentence_all[i * params.batch_size:(i + 1) * params.batch_size + 200].to(device)
        input_user = user_all[i * params.batch_size:(i + 1) * params.batch_size + 100].to(device)
        label = label_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
        match = match_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
        input_timestamp = timestamp_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
        label_logits, label_probs, match_logits,match_probs = model(input_sen, input_user, input_timestamp,params.t_p)
        # loss = criterion(label_logits,label)
        ###################################################
        loss_label = criterion(label_logits, label)
        loss_match = criterion(match_logits, match.reshape(-1))
        loss = (loss_label * first_weight) + loss_match
        running_loss += loss.item()
        ###################################################
        # loss_label = criterion(label_logits, label)
        # loss_match = criterion(match_logits, match.reshape(-1))
        # running_loss += loss_match.item()
        ###################################################
        # match_all_prob.extend(match_probs[:, 1].cpu().numpy())
        # match_all_labels.extend(match)
        match_all_prob.extend(match_probs[:, 1].cpu().detach().numpy())
        for i in range(len(match)):
            match_all_labels.extend(match[i].cpu())
        label_all_prob.extend(label_probs[:, 1].cpu().detach().numpy())
        label_all_labels.extend(label.cpu())
    epoch_loss = running_loss / len(sentence_all)
    total_time = time.time() - time_start
    return total_time,epoch_loss,roc_auc_score(label_all_labels,label_all_prob),roc_auc_score(match_all_labels, match_all_prob)


def test(params, model,sentence_all, user_all, timestamp_all,label_all,match_all,fristQ):
    if fristQ == 0:
        model.eval()
        device = model.device
        N = len(label_all) // params.batch_size
        match_all_prob = []
        match_all_labels = []
        label_all_prob = []
        label_all_labels = []
        time_start = time.time()
        for i in range(N):
            input_sen = sentence_all[i * params.batch_size:(i + 1) * params.batch_size + 200].to(device)
            input_user = user_all[i * params.batch_size:(i + 1) * params.batch_size + 100].to(device)
            label = label_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
            match = match_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
            input_timestamp = timestamp_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
            label_logits, label_probs, match_logits,match_probs = model(input_sen, input_user, input_timestamp,params.t_p)
            match_all_prob.extend(match_probs[:, 1].cpu().detach().numpy())
            for i in range(len(match)):
                match_all_labels.extend(match[i].cpu())
            label_all_prob.extend(label_probs[:, 1].cpu().detach().numpy())
            label_all_labels.extend(label.cpu())
        total_time = time.time() - time_start
        return total_time,roc_auc_score(label_all_labels,label_all_prob),roc_auc_score(match_all_labels,match_all_prob)
    else:
        model.eval()
        device = model.device
        N = len(label_all) // params.batch_size
        match_all_prob = []
        match_all_labels = []
        label_all_prob = []
        label_all_labels = []
        time_start = time.time()
        for i in range(N):
            input_sen = sentence_all[i * params.batch_size:(i + 1) * params.batch_size + 200].to(device)
            input_user = user_all[i * params.batch_size:(i + 1) * params.batch_size + 100].to(device)
            label = label_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
            match = match_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
            input_timestamp = timestamp_all[i * params.batch_size:(i + 1) * params.batch_size].to(device)
            label_logits, label_probs, match_logits, match_probs = model(input_sen, input_user, input_timestamp,
                                                                         params.t_p)
            # match_all_prob.extend(match_probs[:, 1].cpu().detach().numpy())
            for i in range(len(match)):
                if label[i] == 1:
                    match_all_labels.extend(match[i].cpu())
                    match_all_prob.extend(match_probs[100*i:100*i+100, 1].cpu().detach().numpy())
            label_all_prob.extend(label_probs[:, 1].cpu().detach().numpy())
            label_all_labels.extend(label.cpu())
        total_time = time.time() - time_start
        return total_time, roc_auc_score(label_all_labels, label_all_prob), roc_auc_score(match_all_labels,
                                                                                          match_all_prob)

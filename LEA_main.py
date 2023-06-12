import os
import warnings
import torch
import numpy as np
import argparse
import torch.nn as nn
from dice_loss import DiceLoss
from LEA_run import train,valid,test
from LEAGUE_MODEL import LEAGUE
from data import LEA_Dataset
from dataloader import loaddata
# torch.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wd', type=float, default=1e-4, help='the weight decay of optimizer')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--gpu_index', type=int, default=0, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument('--epochs', type=int, default=500, help='number of iterations')
    parser.add_argument('--max_length', type=int, default=50, help='max length of sentences')
    parser.add_argument('--batch_size', type=int, default=64, help='the batch size')
    parser.add_argument('--patience', type=int, default=30, help='for test')
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help='initial learning rate')
    parser.add_argument('--model_type', type=str, default='LEA_MODEL', help='the name of model type')
    parser.add_argument('--first_w', type=float, default=0, help='initial learning rate')
    parser.add_argument('--t_p', type=int, default=0, help='test part 0-normal 1-without label 2-without timestamp')
    parser.add_argument('--m_t', type=int, default=0, help='task match-0 task label-1')
    parser.add_argument('--firstQ', type=int, default=0, help='firstQ 1 other 0')
    parser.add_argument('--dataset_type', type=str, default='bigdata22', help='bigdata22 or bigdata23 or synthetic')
    parser.add_argument("--embedding_dim", default=50)
    parser.add_argument("--lstm_hidden_size", default=50)
    parser.add_argument("--dropout", default=0.5)
    parser.add_argument("--num_classes", default=2)
    parser.add_argument("--loss_func",default='criterion')
    parser.add_argument("--seed", default='1')

    params = parser.parse_args()
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    if params.dataset_type == 'bigdata22':
        parser.add_argument("--user_num", default=61,help='wiki 77 other 61')
        parser.add_argument("--vocabs_size", default=2150)
        vocab_file = 'data/LEA_MODEL/bigdata22_vocab.txt'
        train_file = 'data/LEA_MODEL/bigdata22_train.csv'
        dev_file = 'data/LEA_MODEL/bigdata22_valid.csv'
        test_file = 'data/LEA_MODEL/bigdata22_test.csv'

    elif params.dataset_type == 'bigdata23':
        parser.add_argument("--user_num", default=77)
        parser.add_argument("--vocabs_size", default=1950)
        vocab_file = 'data/LEA_MODEL/bigdata23_vocab.txt'
        train_file = 'data/LEA_MODEL/bigdata_23_train.csv'
        dev_file = 'data/LEA_MODEL/bigdata_23_valid.csv'
        test_file = 'data/LEA_MODEL/bigdata_23_test.csv'

    else:
        parser.add_argument("--user_num", default=61)
        parser.add_argument("--vocabs_size", default=31000)
        vocab_file = 'data/LEA_MODEL/synthetic_vocab.txt'
        train_file = 'data/LEA_MODEL/synthetic_train.csv'
        dev_file = 'data/LEA_MODEL/synthetic_valid.csv'
        test_file = 'data/LEA_MODEL/synthetic_test.csv'

    params = parser.parse_args()


    f = open("./Log/" + params.model_type + '_' + str(params.batch_size) + '_' +  str(params.wd) + '_' +
             str(params.lr) + '_'+ str(params.firstQ) +'_'+ str(params.seed) +'_' + "log.txt", "w")
    #
    f.write(str(params))

    device = torch.device("cuda:{}".format(params.gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = LEA_Dataset(train_file, vocab_file, params.max_length)
    train_sentence, train_user, train_label, train_timestamp, train_match  = loaddata(train_data,params.batch_size)
    print("\t* Loading validation data...")
    val_data = LEA_Dataset(dev_file, vocab_file, params.max_length)
    val_sentence, val_user, val_label, val_timestamp, val_match  = loaddata(val_data,params.batch_size)
    print("\t* Loading test data...")
    test_data = LEA_Dataset(test_file, vocab_file, params.max_length)
    test_sentence, test_user, test_label, test_timestamp, test_match  = loaddata(test_data,params.batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = LEAGUE(params, device=device).to(device)
    # -------------------- Preparation for training  ------------------- #
    if params.loss_func == 'criterion':
        loss_func = nn.CrossEntropyLoss()
    elif params.loss_func == 'diceloss':
        loss_func = DiceLoss(with_logits=True, ohem_ratio=0.1)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=params.lr,weight_decay=params.wd)

    best_score = 0.0
    best_label_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    # Continuing training from a checkpoint if one was gi   ven as argument
    # Compute loss and accuracy before starting (or resuming) training.
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training {} model on device: {}".format(params.model_type,device), 20 * "=")
    patience_counter = 0

    for epoch in range(start_epoch, params.epochs + 1):
        print("* Training epoch {}:".format(epoch))
        f.write("* Training epoch {}:".format(epoch))

        time_train,epoch_loss, label_auc_train,match_auc_train= train(params,model,train_sentence,train_user,
                                                                        train_timestamp,train_label,train_match,
                                                                        optimizer,loss_func,params.max_grad_norm,
                                                                      first_weight=params.first_w)
        print("-> Training time: {:.4f}s, loss = {:.4f},auc_match: {:.4f}\n"
              .format(time_train, epoch_loss,match_auc_train))
        f.write("-> Training time: {:.4f}s, loss = {:.4f},auc_match: {:.4f}\n"
                      .format(time_train, epoch_loss,match_auc_train))
        if params.m_t == 1:
            print("-> Training time: {:.4f}s, loss = {:.4f},auc_label: {:.4f}\n"
                  .format(time_train, epoch_loss,label_auc_train))
            f.write("-> Training time: {:.4f}s, loss = {:.4f},auc_label: {:.4f}\n"
                          .format(time_train, epoch_loss,label_auc_train))

        time_vaild,epoch_loss,label_auc_vaild,match_auc_vaild = valid(params,model,val_sentence,val_user,val_timestamp,
                                                                        val_label,val_match,loss_func,
                                                                      first_weight=params.first_w)
    #

        print("-> Valid. time: {:.4f}s, loss: {:.4f}, auc: {:.4f}\n"
              .format(time_vaild, epoch_loss, match_auc_vaild))
        f.write("-> Valid. time: {:.4f}s, loss: {:.4f}, auc: {:.4f}\n"
              .format(time_vaild, epoch_loss, match_auc_vaild))
        if params.m_t == 1:
            print("-> Valid. time: {:.4f}s, loss: {:.4f}, label_auc: {:.4f}\n"
                  .format(time_vaild, epoch_loss, label_auc_vaild))
            f.write("-> Valid. time: {:.4f}s, loss: {:.4f}, label_auc: {:.4f}\n"
                  .format(time_vaild, epoch_loss, label_auc_vaild))

        time_test,label_auc_test,match_auc_test = test(params,model,test_sentence,test_user,test_timestamp,
                                                           test_label,test_match,params.firstQ)

        print("->test. time: {:.4f}s, test auc: {:.4f}\n".format(time_test,match_auc_test))
        f.write("->test. time: {:.4f}s, test auc: {:.4f}\n".format(time_test,match_auc_test))
        if params.m_t == 1:
            print("->test. time: {:.4f}s, label_test auc: {:.4f}\n".format(time_test,label_auc_test))
            f.write("->test. time: {:.4f}s, label_test auc: {:.4f}\n".format(time_test,label_auc_test))

    #     # Update the optimizer's learning rate with the scheduler.
    #
    #     # Early stopping on validation accuracy.
        if match_auc_test <= best_score:
            patience_counter += 1
        else:
            best_score = match_auc_test
            best_label_score = label_auc_test
            patience_counter = 0
    #
        if patience_counter >= params.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            f.write("-> Early stopping: patience limit reached, stopping...")
            break


    print("->best test_auc : {:.4f}\n".format(best_score))
    f.write("->best test_auc : {:.4f}\n".format(best_score))
    if params.m_t == 1:
        print("->best test_label_auc : {:.4f}\n".format(best_label_score))
        f.write("->best test_label_auc : {:.4f}\n".format(best_label_score))
    f.close()



if __name__ == '__main__':
    main()






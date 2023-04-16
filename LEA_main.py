import os
import warnings
import torch
import numpy as np
import argparse
import torch.nn as nn
from dice_loss import DiceLoss
from LEA_run import train
from LEAGUE_MODEL import LEAGUE
from data import LEA_Dataset
from dataloader import loaddata
# torch.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wd', type=float, default=1e-4, help='the weight decay of optimizer')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--gpu_index', type=int, default=0, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='initial learning rate')
    parser.add_argument('--max_length', type=int, default=50, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='initial learning rate')
    parser.add_argument('--patience', type=int, default=5, help='initial learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help='initial learning rate')
    parser.add_argument('--model_type', type=str, default='LEA_MODEL', help='initial learning rate')
    parser.add_argument("--embedding_dim", default=50)
    parser.add_argument("--lstm_hidden_size", default=50)
    parser.add_argument("--user_num", default=61)
    parser.add_argument("--dropout", default=0.5)
    parser.add_argument("--vocabs_size", default=2500)
    parser.add_argument("--prediction", default='full')
    parser.add_argument("--num_classes", default=2)
    parser.add_argument("--loss_func",default='criterion')
    parser.add_argument("--seed", default='1')

    params = parser.parse_args()
    torch.manual_seed(params.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(params.seed)  # 为当前GPU设置随机种子

    vocab_file = 'vocab.txt'
    train_file = 'new_train.csv'
    dev_file = 'new_valid.csv'
    test_file = 'new_test.csv'

    # f = open("./Log/" +  params.model_type + '/' +  params.model_type + '_' + params.prediction + '_' + params.loss_func + '_' + str(params.wd) + '_'
    #          + str(params.lr) + '_' + str(params.dropout) +'_' + 'seed' + str(params.seed) +
    #          '_' + "log.txt", "w")
    #
    # f.write(str(params))

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
    test_sentence, test_user, test_label, test_timestamp, test_match  = loaddata(val_data,params.batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = LEAGUE(params, device=device).to(device)
    # -------------------- Preparation for training  ------------------- #
    if params.loss_func == 'criterion':
        loss_func = nn.CrossEntropyLoss()
    elif params.loss_func == 'diceloss':
        loss_func = DiceLoss(with_logits=True, ohem_ratio=0.1)
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    optimizer = torch.optim.Adam(parameters, lr=params.lr,weight_decay=params.wd)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    # Continuing training from a checkpoint if one was gi   ven as argument
    # Compute loss and accuracy before starting (or resuming) training.
    # _, valid_loss, valid_accuracy, auc = validate(model, dev_loader, criterion,model_type=params.model_type)
    # _, valid_loss, valid_accuracy, auc = valid(model,val_sentence,val_user,val_timestamp,val_label,val_match
    #                                            , loss_func,model_type=params.model_type)
    # print("\t* Validation loss before training: {:.4f},auc: {:.4f}".format(valid_loss,auc))
    # f.write("\t* Validation loss before training: {:.4f},auc: {:.4f}".format(valid_loss,auc))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training {} model on device: {}".format(params.model_type,device), 20 * "=")
    patience_counter = 0

    for epoch in range(start_epoch, params.epochs + 1):
        print("* Training epoch {}:".format(epoch))
        # f.write("* Training epoch {}:".format(epoch))

        # epoch_time, epoch_loss, epoch_accuracy, epoch_auc_train = train(model, train_loader, optimizer,
        #                                                                 criterion, epoch, params.max_grad_norm,
        #                                                                 model_type=params.model_type)
        epoch_loss, label_auc_train ,match_auc_train = train(params,model,train_sentence,train_user,
                                                                        train_timestamp,train_label,train_match,
                                                                        optimizer,loss_func,params.max_grad_norm)
        print("-> Training time: {:.4f}s, loss = {:.4f},auc_label: {:.4f},auc_match: {:.4f}\n"
              .format(0, epoch_loss,label_auc_train,match_auc_train))

        # epoch_loss, label_auc_train  = train(params,model,train_sentence,train_user,
        #                                                                 train_timestamp,train_label,train_match,
        #                                                                 optimizer,loss_func,params.max_grad_norm)
        # print("-> Training time: {:.4f}s, loss = {:.4f},auc_label: {:.4f},auc_match: {:.4f}\n"
        #       .format(0, epoch_loss,label_auc_train,0))


    #     # f.write("-> Training time: {:.4f}s, loss = {:.4f},auc: {:.4f}\n"
    #     #       .format(epoch_time, epoch_loss, epoch_auc_train))
    #
    # #     print("* Validation for epoch {}:".format(epoch))
    # #     # f.write("* Validation for epoch {}:".format(epoch))
    # #
    #     epoch_loss,epoch_auc_vaild = valid(params,model,val_sentence,val_user,val_timestamp,
    #                                                                     val_label,val_match,loss_func)
    # #
    #     print("-> Valid. time: {:.4f}s, loss: {:.4f}, auc: {:.4f}\n"
    #           .format(0, epoch_loss, epoch_auc_vaild))
    # #     # f.write("-> Valid. time: {:.4f}s, loss: {:.4f}, auc: {:.4f}\n"
    # #     #         .format(epoch_time, epoch_loss, epoch_auc_vaild))
    # #
    #     epoch_auc_test = test(params,model,test_sentence,test_user,test_timestamp,test_label,test_match)
    #
    #     print("->test auc: {:.4f}\n".format(epoch_auc_test))
    # #     # f.write("->test auc: {:.4f}\n".format(epoch_auc_test))
    # #     # Update the optimizer's learning rate with the scheduler.
    # #
    # #     scheduler.step(epoch_auc_train)
    # #     # Early stopping on validation accuracy.
    # #     if epoch_auc_test <= best_score:
    # #         patience_counter += 1
    # #     else:
    # #         best_score = epoch_auc_test
    # #         patience_counter = 0
    # #
    # #     if patience_counter >= params.patience:
    # #         print("-> Early stopping: patience limit reached, stopping...")
    # #         # f.write("-> Early stopping: patience limit reached, stopping...")
    # #         break
    # # print("->best test_auc : {:.4f}\n".format(best_score))
    # # # f.write("->best test_auc : {:.4f}\n".format(best_score))
    # # # f.close()
    #


if __name__ == '__main__':
    main()




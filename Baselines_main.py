import os
import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from models.data import LCQMC_Dataset, load_embeddings
from Baselines_run import train, validate,test
from dice_loss import DiceLoss
from models.AP_BILSTM import AP_BILSTM
from models.ESIM.model import ESIM
from models.cnn_bilstm import CNN_BILSTM
from models.Bi_LSTM import BI_LSTM
from models.CNN import CNN
from models.AP_CNN import AP_CNN
from models.ABCNN import ABCNN

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wd', type=float, default=5e-5, help='the weight decay of optimizer')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--gpu_index', type=int, default=0, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='initial learning rate')
    parser.add_argument('--max_length', type=int, default=50, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='initial learning rate')
    parser.add_argument('--patience', type=int, default=5, help='initial learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help='initial learning rate')
    parser.add_argument('--model_type', type=str, default='CNN', help='initial learning rate')
    parser.add_argument('--dataset_type', type=str, default='bigdata22', help='initial learning rate')
    parser.add_argument('--with_label', type=int, default=0, help='initial learning rate')
    parser.add_argument("--dropout", default=0.5)
    parser.add_argument("--prediction", default='full')
    parser.add_argument("--num_classes", default=2)
    parser.add_argument("--loss_func",default='criterion')
    parser.add_argument("--seed", default='1')
    params = parser.parse_args()

    if params.model_type == 'ESIM':
        parser.add_argument("--embedding_dim", default=50)
        parser.add_argument("--hidden_size", default=150)

    elif params.model_type == 'CNN-BILSTM':
        parser.add_argument("--hidden_size", default=50)

    elif params.model_type == 'CNN':
        parser.add_argument("--embedding_dim", default=50)
        parser.add_argument("--hidden_size", default=50)
        parser.add_argument("--out_channels", default=50)
        parser.add_argument("--kernel_sizes", default=[3])
        parser.add_argument("--padding", default=1)

    elif params.model_type == 'BI-LSTM':
        parser.add_argument("--embedding_dim", default=50)
        parser.add_argument("--hidden_size", default=50)

    elif params.model_type == 'AP-CNN':
        parser.add_argument("--embedding_dim", default=50)
        parser.add_argument("--hidden_size", default=50)
        parser.add_argument("--out_channels", default=50)
        parser.add_argument("--init_U", default='randn')
        parser.add_argument("--kernel_sizes", default=[3])
        parser.add_argument("--padding", default=1)

    elif params.model_type == 'AP-BILSTM':
        parser.add_argument("--embedding_dim", default=50)
        parser.add_argument("--hidden_size", default=100)
        parser.add_argument("--lstm_hidden_size", default=50)
        parser.add_argument("--init_U", default='randn')

    elif params.model_type == 'ABCNN':
        parser.add_argument("--embedding_dim", default=50)
        parser.add_argument("--hidden_size", default=100)
        parser.add_argument("--num_layer", default=1)
        parser.add_argument("--linear_size", default=300)

    params = parser.parse_args()

    if params.dataset_type == 'bigdata22':
        parser.add_argument("--vocabs_size", default=2150)
        train_file = 'data/baseline_data/bigdata22_train.csv'
        vocab_file = 'data/baseline_data/bigdata22_vocab.txt'
        dev_file = 'data/baseline_data/bigdata22_valid.csv'
        if params.with_label == 1:
            test_file = 'data/baseline_data/bigdata22_test_with_label.csv'
        else:
            test_file = 'data/baseline_data/bigdata22_test_without_label.csv'
        embeddings_file = 'data/baseline_data/w2v.txt'

    elif params.dataset_type == 'bigdata23':
        parser.add_argument("--vocabs_size", default=1950)
        train_file = 'data/baseline_data/bigdata23_train.csv'
        vocab_file = 'data/baseline_data/bigdata23_vocab.txt'
        dev_file = 'data/baseline_data/bigdata23_valid.csv'
        if params.with_label == 1:
            test_file = 'data/baseline_data/bigdata23_test_with_label.csv'
        else:
            test_file = 'data/baseline_data/bigdata23_test_without_label.csv'
        embeddings_file = 'data/baseline_data/w2v.txt'

    else:
        parser.add_argument("--vocabs_size", default=32000)
        train_file = 'data/baseline_data/synthetic_train.csv'
        vocab_file = 'data/baseline_data/synthetic_vocab.txt'
        dev_file = 'data/baseline_data/synthetic_valid.csv'
        if params.with_label == 1:
            test_file = 'data/baseline_data/synthetic_test_with_label.csv'
        else:
            test_file = 'data/baseline_data/synthetic_test_without_label.csv'
        embeddings_file = 'data/baseline_data/w2v.txt'

    params = parser.parse_args()
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)

    os.makedirs("./Log/" + params.model_type, exist_ok=True)

    f = open("./Log/" +  params.model_type + '/' +  params.model_type + '_' + params.prediction + '_' + params.loss_func + '_' + str(params.wd) + '_'
             + str(params.lr) + '_' + str(params.dropout) +'_' + 'seed' + str(params.seed) +
             '_' + "log.txt", "w")
    # f = open("./Log/" +  params.model_type + '_' + params.prediction + '_' + params.loss_func + '_' + str(params.wd) + '_'
    #          + str(params.lr) + '_' + str(params.dropout) +
    #          '_' + "log.txt", "w")
    f.write(str(params))

    device = torch.device("cuda:{}".format(params.gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = LCQMC_Dataset(train_file, vocab_file, params.max_length)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=params.batch_size)
    print("\t* Loading validation data...")
    dev_data = LCQMC_Dataset(dev_file, vocab_file, params.max_length)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=params.batch_size)
    print("\t* Loading test data...")
    test_data = LCQMC_Dataset(test_file, vocab_file, params.max_length)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=params.batch_size)
    # -------------------- Model definition ------------------- #
    # print("\t* Building model...")
    # if params.model_type == 'RE2':
    #     embeddings = load_embeddings(embeddings_file)
    #     model = RE2(params, embeddings, device=device).to(device)
    print("\t* Building model...")
    if params.model_type == 'ESIM':
        embeddings = load_embeddings(embeddings_file)
        model = ESIM(hihdden_size=params.hidden_size,embeddings=embeddings,dropout=params.dropout,device=device).to(device)
    elif params.model_type == 'CNN-BILSTM':
        embeddings = load_embeddings(embeddings_file)
        model = CNN_BILSTM(params, embeddings, device=device).to(device)
    elif params.model_type == 'BI-LSTM':
        model = BI_LSTM(params,device=device).to(device)
    elif params.model_type == 'CNN':
        model = CNN(params,device=device).to(device)
    elif params.model_type == 'AP-CNN':
        model = AP_CNN(params,device=device).to(device)
    elif params.model_type == 'AP-BILSTM':
        model = AP_BILSTM(params, device=device).to(device)
    elif params.model_type == 'ABCNN':
        embeddings = load_embeddings(embeddings_file)
        model = ABCNN(params,embeddings,device=device).to(device)
    # -------------------- Preparation for training  ------------------- #
    if params.loss_func == 'criterion':
        loss_func = nn.CrossEntropyLoss()
    elif params.loss_func == 'diceloss':
        loss_func = DiceLoss(with_logits=True, ohem_ratio=0.1)
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
    _, valid_loss, valid_accuracy, auc = validate(model, dev_loader, loss_func,model_type=params.model_type)
    print("\t* Validation loss before training: {:.4f},auc: {:.4f}".format(valid_loss,auc))
    f.write("\t* Validation loss before training: {:.4f},auc: {:.4f}".format(valid_loss,auc))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training {} model on device: {}".format(params.model_type,device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, params.epochs + 1):
        print("* Training epoch {}:".format(epoch))
        f.write("* Training epoch {}:".format(epoch))

        # epoch_time, epoch_loss, epoch_accuracy, epoch_auc_train = train(model, train_loader, optimizer,
        #                                                                 criterion, epoch, params.max_grad_norm,
        #                                                                 model_type=params.model_type)
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc_train = train(model, train_loader, optimizer,
                                                                        loss_func, epoch, params.max_grad_norm,
                                                                        model_type=params.model_type)


        print("-> Training time: {:.4f}s, loss = {:.4f},auc: {:.4f}\n"
              .format(epoch_time, epoch_loss,epoch_auc_train))
        f.write("-> Training time: {:.4f}s, loss = {:.4f},auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, epoch_auc_train))

        print("* Validation for epoch {}:".format(epoch))
        f.write("* Validation for epoch {}:".format(epoch))

        # epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, dev_loader, criterion,
        #                                                              model_type=params.model_type)
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc_vaild = validate(model, dev_loader, loss_func,
                                                                     model_type=params.model_type)

        print("-> Valid. time: {:.4f}s, loss: {:.4f}, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, epoch_auc_vaild))
        f.write("-> Valid. time: {:.4f}s, loss: {:.4f}, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, epoch_auc_vaild))

        batch_time, total_time, accuracy, epoch_auc_test = test(model, test_loader,model_type=params.model_type)
        print("->test auc: {:.4f}\n".format(epoch_auc_test))
        f.write("->test auc: {:.4f}\n".format(epoch_auc_test))
        # Update the optimizer's learning rate with the scheduler.

        scheduler.step(epoch_auc_train)

        # Early stopping on validation accuracy.
        if epoch_auc_test <= best_score:
            patience_counter += 1
        else:
            best_score = epoch_auc_test
            patience_counter = 0

        if patience_counter >= params.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            f.write("-> Early stopping: patience limit reached, stopping...")
            break
    print("->best test_auc : {:.4f}\n".format(best_score))
    f.write("->best test_auc : {:.4f}\n".format(best_score))
    f.close()

if __name__ == "__main__":
    main()


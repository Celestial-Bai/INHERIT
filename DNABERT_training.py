# Author: Zeheng Bai
##### IGNITER TRAINING TRANSFORMER MODEL #####
from basicsetting import *
from readfasta import *
from DNABERTModels import *
from Dataset_config import *
from IGN_config import *
import math
import matplotlib.pyplot as plt

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("--bertdir",
                        type=str,
                        default='')
    PARSER.add_argument("--checkpoint",
                        type=str,
                        default = '')
    PARSER.add_argument("--outdir",
                        type=str,
                        required=True)

    args = PARSER.parse_args()

    ##### Hyperparamters: in Network_config #####
    pid = 'checkpoints_DNABERT'
    path = os.getcwd() + "/" + str(pid)
    if os.path.exists(path) is False:
        os.makedirs(path)
    config = BertConfig.from_pretrained(CONFIG_PATH)
    early_stopping = EarlyStopping_acc(checkpoint = pid + '/' + 'bestacc_' + args.outdir, patience=PATIENCE, verbose=True)
    tokenizer = DNATokenizer.from_pretrained(CONFIG_PATH)
    X_bac_tr = read_fasta(open(BAC_TR_PATH), KMERS, SEGMENT_LENGTH)
    X_pha_tr = read_fasta(open(PHA_TR_PATH), KMERS, SEGMENT_LENGTH)
    X_train = X_pha_tr + X_bac_tr
    y_train = torch.cat((torch.ones(len(X_pha_tr)), torch.zeros(len(X_bac_tr))))
    y_train = y_train.to(torch.long).unsqueeze(1)
    X_bac_val = read_fasta(open(BAC_VAL_PATH), KMERS, SEGMENT_LENGTH)
    X_pha_val = read_fasta(open(PHA_VAL_PATH), KMERS, SEGMENT_LENGTH)
    X_val = X_pha_val + X_bac_val
    y_val = torch.cat((torch.ones(len(X_pha_val)), torch.zeros(len(X_bac_val))))
    y_val = y_val.to(torch.long).unsqueeze(1)
    train_data = IGNDataset(X_seq=X_train, y=y_train, tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=TR_BATCHSIZE, shuffle=True, num_workers=TR_WORKERS)
    val_data = IGNDataset(X_seq=X_val, y=y_val, tokenizer=tokenizer)
    val_loader = DataLoader(val_data, batch_size=VAL_BATCHSIZE, shuffle=True, num_workers=VAL_WORKERS)
    bertmodel = Baseline_BERT(freeze_bert=False, config=config, bert_dir = args.bertdir)
    bert_params = list(map(id, bertmodel.bert.parameters()))
    new_params = filter(lambda p: id(p) not in bert_params, bertmodel.parameters())
    opt = torch.optim.Adam([{'params': bertmodel.bert.parameters(), 'lr': LEARNING_RATE},
                                  {'params': new_params}], lr=LEARNING_RATE)
    if torch.cuda.device_count() > 1:
        bertmodel = torch.nn.DataParallel(bertmodel)
    if args.checkpoint != '':
        sdict = torch.load(args.checkpoint)
        bertmodel.load_state_dict(sdict)
    bertmodel.to(device)
    sigmoid = torch.nn.Sigmoid()
    loss_func = torch.nn.BCEWithLogitsLoss()
    print("The checkpoints will be in: " + path)
    print("start to train")
    train_loss_value=[]
    train_acc_value=[]
    val_loss_value=[]
    val_acc_value=[]
    batchsize = TR_BATCHSIZE
    for epoch in range(EPOCHS):
        running_loss = 0.0
        valrun_loss = 0.0
        sum_correct = 0
        sum_total = 0
        t = len(train_loader.dataset)
        bertmodel.train()
        with tqdm(total=100) as pbar:
            for i, (x, y) in enumerate(train_loader):
                X_seq = BatchEncoding(x)
                X_seq = X_seq.to(device)
                Label = Variable(y)
                Label = Label.squeeze(0)
                Label = Label.to(device)
                opt.zero_grad()
                out = bertmodel(input_ids=X_seq['input_ids'].squeeze(1), token_type_ids=X_seq['token_type_ids'].squeeze(1), attention_mask=X_seq['attention_mask'].squeeze(1))
                loss = loss_func(out, Label.to(torch.float32))
                loss.backward()
                opt.step()
                # print statistics
                running_loss += loss.item()
                out = sigmoid(out)
                predicted = torch.round_(out)
                sum_total += Label.size(0)
                sum_correct += (predicted == Label).sum().item()
                pbar.update(100 * batchsize / t)
            pbar.close()
        print("epochs={}, mean loss={}, accuracy={}"
            .format(epoch + 1, running_loss * batchsize / t, float(sum_correct / sum_total)))
        train_loss_value.append(running_loss * batchsize / t)
        train_acc_value.append(float(sum_correct / sum_total))
        bertmodel.eval()
        t = len(val_loader.dataset)
        for i, (x, y) in enumerate(val_loader):
            X_seq = BatchEncoding(x)
            X_seq = X_seq.to(device)
            Label = Variable(y)
            Label = Label.squeeze(0)
            Label = Label.to(device)
            val_output = bertmodel(input_ids=X_seq['input_ids'].squeeze(1), token_type_ids=X_seq['token_type_ids'].squeeze(1), attention_mask=X_seq['attention_mask'].squeeze(1))
            loss = loss_func(val_output, Label.to(torch.float32))
            valrun_loss += loss.item()
            val_output = sigmoid(val_output)
            predicted = torch.round_(val_output)
            sum_total += Label.size(0)
            sum_correct += (predicted == Label).sum().item()
        val_acc = float(sum_correct / sum_total)
        print("epochs={}, val loss={}, val accuracy={}"
            .format(epoch + 1, valrun_loss * batchsize / t, float(sum_correct / sum_total)))
        val_loss_value.append(valrun_loss * batchsize / t)
        val_acc_value.append(float(sum_correct / sum_total))
        plt.cla()
        plt.plot(range(1,len(train_loss_value) + 1), train_loss_value, marker='o', mec='r', mfc='w', label=u'Training Loss')
        plt.plot(range(1,len(train_loss_value) + 1), val_loss_value, marker='*', ms=10, label=u'Validation Loss')
        plt.legend()
        plt.xticks(range(1,len(train_loss_value) + 1), range(1,len(train_loss_value) + 1), rotation=45)
        plt.margins(0)
        plt.xlabel(u"Epochs")
        plt.ylabel("Loss")
        plt.title("Loss plot")
        plt.savefig("Lossplot_new.png")
        plt.cla()
        plt.plot(range(1,len(train_acc_value) + 1), train_acc_value, marker='o', mec='r', mfc='w', label=u'Training Accuracy')
        plt.plot(range(1,len(train_acc_value) + 1), val_acc_value, marker='*', ms=10, label=u'Validation Accuracy')
        plt.legend()
        plt.ylim(0.4, 1)
        plt.xticks(range(1,len(train_acc_value) + 1), range(1,len(train_acc_value) + 1), rotation=45)
        plt.margins(0)
        plt.xlabel(u"Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy plot")
        plt.savefig("Accplot_new.png")
        torch.save(bertmodel.state_dict(), pid + '/' + 'checkpoint_' + str(epoch) + '_' + args.outdir)
        early_stopping(val_acc, bertmodel)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(bertmodel.state_dict(), args.outdir)

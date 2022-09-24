# Author: Zeheng Bai
##### INHERIT FOR PREDICTION #####
# Device: GPU #
from basicsetting import *
from readfasta import *
from INHERITModels import *
from Dataset_config import *
from IHT_config import *
import math

def prediction(input_sequence, tokenizer, bertmodel, batch_size=TR_BATCHSIZE):
    with torch.no_grad():
        sigmoid = torch.nn.Sigmoid()
        batchsize = batch_size
        X_test_dataloader = DataLoader(input_sequence, batch_size=batchsize, shuffle=False, num_workers = TR_WORKERS)
        preds = torch.empty((0, 1))
        for i, seq in enumerate(X_test_dataloader):
            X_seq = tokenizer(seq, return_tensors="pt")
            X_seq = X_seq.to(device)
            pred = bertmodel(input_ids=X_seq['input_ids'], token_type_ids=X_seq['token_type_ids'], attention_mask=X_seq['attention_mask'])
            pred = pred.detach().cpu()
            pred = sigmoid(pred)
            preds = torch.cat((preds, pred))
    return preds

def pred_fasta(fasta_path, kmer, segment_length, threshold):
    fasta_dic = segment_fasta(fasta_path, segment_length)
    outputfile = open(outfile, "a")
    for item_id in fasta_dic:
        matrices = []
        for seq in fasta_dic[item_id]:
            seq = seq.upper()

            if re.match('^[ACGTN]+$', str(seq)) is None:
                continue

            matrix = seq2kmer(str(seq), kmer)
            matrices.append(matrix)
        name = item_id
        print(name)
        print(len(matrices))
        scores = prediction(input_sequence=matrices, tokenizer=tokenizer, bertmodel=bertmodel)
        print(scores)
        mean_score = float(sum(scores) / len(scores))
        if mean_score >= threshold:
            category = "Phage"
        else:
            category = "Bacteria"
        outputfile.write('\t'.join([name, category, str(mean_score)]) + '\n')
    outputfile.close()
    print("Done. Thank you for using our model.")


if __name__ == '__main__':
    ######### Add Parser #############
    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("--sequence",
                        type=argparse.FileType('r'),
                        required=True)
    PARSER.add_argument("--withpretrain",
                        type=str,
                        required=True)
    PARSER.add_argument("--model",
                        type=str,
                        required=True)               
    PARSER.add_argument("--out",
                        type=str,
                        required=True)
    args = PARSER.parse_args()

    ######### Load Model #########
    config = BertConfig.from_pretrained(CONFIG_PATH)
    if args.withpretrain == "True":
        bertmodel = Baseline_IHT(freeze_bert=True, config=config, bac_bert_dir=BAC_PTRMODEL, pha_bert_dir=PHA_PTRMODEL)
    elif args.withpretrain == "False":
        bertmodel = Baseline_DNABERT(freeze_bert=True, config=config)
    tokenizer = DNATokenizer.from_pretrained(CONFIG_PATH)
    #if torch.cuda.device_count() > 1:
    bertmodel = torch.nn.DataParallel(bertmodel)
    sdict = torch.load(args.model)
    bertmodel.load_state_dict(sdict)
    bertmodel.eval()
    bertmodel = bertmodel.to(device)
    outfile = args.out
    outputfile = open(outfile, "w")
    outputfile.write('\t'.join(['name', 'category', 'score']) + '\n')
    outputfile.close()
    pred_fasta(fasta_path=args.sequence, kmer=KMERS, segment_length=SEGMENT_LENGTH, threshold=THRESHOLD)
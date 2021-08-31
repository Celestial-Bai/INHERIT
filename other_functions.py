# Author: Zeheng Bai
##### INHERIT GENERATE TRAINING AND VALIDATION SETS AND TEST THE SHAPE TO BE BALANCED #####
##### MANY FUNCTIONS WILL BE PROPOSED HERE #####
from basicsetting import *
from readfasta import read_fasta, read_from_file_with_enter, write_fasta_cleaned


def shapetest(path):
    Seqs = read_fasta(open(path, 'r'), 5, 500)
    return len(Seqs)


def delete_sequences(path, k):
    Seq_dict = read_from_file_with_enter(path)
    samples = random.sample(Seq_dict.keys(), k)
    for samp in samples:
        del Seq_dict[samp]
    return Seq_dict

def trainval_division(bacpath, phapath):
    X_bac = read_from_file_with_enter(bacpath)
    X_pha = read_from_file_with_enter(phapath)
    for i in range(1000):
        bac_samples = random.sample(X_bac.keys(), int(len(X_bac)/5))
        pha_samples = random.sample(X_pha.keys(), int(len(X_pha)/5))
        bac_valset = {}
        pha_valset = {}
        for bac_sample in bac_samples:
            bac_valset[bac_sample] = X_bac[bac_sample]
        for pha_sample in pha_samples:
            pha_valset[pha_sample] = X_pha[pha_sample]
        bac_trainset = X_bac
        pha_trainset = X_pha
        for bac_sample in bac_samples:
            del bac_trainset[bac_sample]
        for pha_sample in pha_samples:
            del pha_trainset[pha_sample]
        write_fasta_cleaned(seqdict=bac_trainset, outdir="bac_training_1.txt", banned_list=[])
        write_fasta_cleaned(seqdict=pha_trainset, outdir="pha_training_1.txt", banned_list=[])
        write_fasta_cleaned(seqdict=bac_valset, outdir="bac_val_1.txt", banned_list=[])
        write_fasta_cleaned(seqdict=pha_valset, outdir="pha_val_1.txt", banned_list=[])
        bac_train_len = shapetest("bac_training_1.txt")
        pha_train_len = shapetest("pha_training_1.txt")
        bac_val_len = shapetest("bac_val_1.txt")
        pha_val_len = shapetest("pha_val_1.txt")
        if (bac_val_len > pha_val_len) and (bac_train_len > pha_train_len):
            print("Balanced!")
            break
        else:
            print("Not balanced, continue.")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("--mode",
                        type=str,
                        required=True)
    PARSER.add_argument("--sequence",
                        type=str)
    PARSER.add_argument("--bacpath",
                        type=str)
    PARSER.add_argument("--phapath",
                        type=str)
    PARSER.add_argument("--bannedlist",
                        type=argparse.FileType('r'))
    PARSER.add_argument("--k",
                        type=int,
                        default=5)
    PARSER.add_argument("--out",
                        type=str)

    args = PARSER.parse_args()

    if args.mode == 'random_delete':
        banned_list = []
        with args.bannedlist as f:
            for line in f.readlines():
                line = line.strip()
                banned_list.append(line)
        print("read banned list has been done.")
        deleted = delete_sequences(path=args.sequence, k=args.k)
        write_fasta_cleaned(seqdict=deleted, banned_list=banned_list, outdir=args.out)
    if args.mode == 'shape_test':
        shapes = shapetest(path=args.sequence)
        print(shapes)
    if args.mode == 'set_generation':
        trainval_division(bacpath=args.bacpath, phapath=args.phapath)

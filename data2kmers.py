#Author: Zeheng Bai
##### IGNITER GENERATE FASTA SEQUENCE FILE TO K-MERS FILE #####
from basicsetting import *
from readfasta import read_fasta

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("--sequence",
                        type=argparse.FileType('r'),
                        required=True)
    PARSER.add_argument("--out",
                        type=str,
                        required=True)

    args = PARSER.parse_args()

    X_seq = read_fasta(args.sequence, 6, 500)

    outputfile = open(args.out, "w")

    for i in range(len(X_seq)):
        print(X_seq[i], file=outputfile)
    outputfile.close()
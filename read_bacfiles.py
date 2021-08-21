#Author: Zeheng Bai
##### IGNITER GENERATE BACTERIA PRETRAINING & TRAINING SETS #####
# Device: CPU #
from basicsetting import *
from readfasta import read_from_file_with_enter

def read_bacteria_samples(bac_path, k):
    path = os.getcwd() + "/" + bac_path
    foldlists = os.listdir(path)
    samples = random.sample(range(len(foldlists)), k)
    f = open("samplelist_"+ args.out, "w")
    for i in range(len(samples)):
        print(samples[i], file=f)
    f.close()
    X_bac = []
    names = []
    for i in samples:
        pathnew = path + "/" + foldlists[i]
        filelists = os.listdir(pathnew)
        if filelists[0] == "MD5SUMS":
            filename = filelists[1]
        else:
            filename = filelists[0]
        bac_seq_dict = read_from_file_with_enter(pathnew + "/" + filename)
        for i in range(len(bac_seq_dict)):
            X_bac.append(list(bac_seq_dict.values())[i])
            names.append(list(bac_seq_dict.keys())[i])
    return X_bac, names

def write_bac_fasta(bacoutdir, X_bac, names, banned_list):
    f = open(bacoutdir, "w")
    for i in range(len(names)):
        if (names[i].split('.')[0] in banned_list) is False:
            print('>' + names[i], file=f)
        print(X_bac[i], file=f)
    f.close()
    print("write fasta has been done.")

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description=__doc__)
    PARSER.add_argument("--bannedlist",
                        type=argparse.FileType('r'),
                        required=True)
    PARSER.add_argument("--bacpath",
                        type=str,
                        required=True)
    PARSER.add_argument("--k",
                        type=int,
                        default=2000)
    PARSER.add_argument("--out",
                        type=str,
                        required=True)
    args = PARSER.parse_args()

    banned_list = []
    with args.bannedlist as f:
        for line in f.readlines():
            line = line.strip()
            banned_list.append(line)
    print("read banned list has been done.")

    X_bac, names = read_bacteria_samples(bac_path=args.bacpath, k=args.k)
    write_bac_fasta(bacoutdir=args.out, banned_list=banned_list)

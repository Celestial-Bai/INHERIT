#Author: Zeheng Bai
##### INHERIT GENERATE PHAGE PRETRAINING & TRAINING SETS #####
# Device: CPU #
from basicsetting import *
from readfasta import read_from_file_with_enter, write_fasta_seq_names

def write_phage_pretrain_fasta(outdir, names, banned_list):
    f = open(outdir, "w")
    for i in range(len(names)):
        if (names[i].split('.')[0] in banned_list) is False:
            if (names[i] in phage_selfbuilt_ids1) is True:
                print('>' + names[i], file=f)
                print(phage_selfbuilt[phage_selfbuilt_dict[names[i]]], file=f)
            elif (names[i] in phage_seeker_training_set1_ids) is True:
                print('>' + names[i], file=f)
                print(phage_seeker_training_set1[names[i]], file=f)
            elif (names[i] in phage_seeker_training_set2_ids) is True:
                print('>' + names[i], file=f)
                print(phage_seeker_training_set2[names[i]], file=f)
            elif (names[i] in phage_seeker_val_set_ids) is True:
                print('>' + names[i], file=f)
                print(phage_seeker_val_set[names[i]], file=f)
            elif (names[i] in phage_seeker_test_set_ids) is True:
                print('>' + names[i], file=f)
                print(phage_seeker_test_set[names[i]], file=f)
            elif (names[i] in phage_vibrant_set_ids1) is True:
                print('>' + names[i], file=f)
                print(phage_vibrant_set[phage_vibrant_dict[names[i]]], file=f)
    f.close()
    print("write fasta has been done.")

def write_phage_training_fasta(outdir, names, banned_list):
    f = open(outdir, "w")
    for i in range(len(names)):
        if (names[i].split('.')[0] in banned_list) is False:
            if (names[i] in phage_seeker_training_set1_ids) is True:
                print('>' + names[i], file=f)
                print(phage_seeker_training_set1[names[i]], file=f)
            elif (names[i] in phage_seeker_training_set2_ids) is True:
                print('>' + names[i], file=f)
                print(phage_seeker_training_set2[names[i]], file=f)
            elif (names[i] in phage_seeker_val_set_ids) is True:
                print('>' + names[i], file=f)
                print(phage_seeker_val_set[names[i]], file=f)
            elif (names[i] in phage_seeker_test_set_ids) is True:
                print('>' + names[i], file=f)
                print(phage_seeker_test_set[names[i]], file=f)
            elif (names[i] in phage_vibrant_set_ids1) is True:
                print('>' + names[i], file=f)
                print(phage_vibrant_set[phage_vibrant_dict[names[i]]], file=f)
    f.close()
    print("write fasta has been done.")

def split_ids(id):
    return id.split('.')[0]

if __name__ == '__main__':
    phage_selfbuilt = read_from_file_with_enter('/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zhang/working/2021_p1_transformer_virome/bai_new/dataset/Self_built/phage/phage_supp1.fasta')
    phage_selfbuilt_ids = list(phage_selfbuilt.keys())
    phage_selfbuilt_ids1 = list(map(split_ids, phage_selfbuilt_ids))
    phage_selfbuilt_dict = {}
    for i in range(len(phage_selfbuilt_ids1)):
        phage_selfbuilt_dict[phage_selfbuilt_ids1[i]] = phage_selfbuilt_ids[i]
    phage_seeker_training_set1 = read_from_file_with_enter('/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zhang/working/2021_p1_transformer_virome/bai_new/dataset/Seeker/training/PHAGETR1_FF.txt')
    phage_seeker_training_set1_ids = list(phage_seeker_training_set1.keys())
    phage_seeker_training_set2 = read_from_file_with_enter('/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zhang/working/2021_p1_transformer_virome/bai_new/dataset/Seeker/training/PHAGETR2_FF.txt')
    phage_seeker_training_set2_ids = list(phage_seeker_training_set2.keys())
    phage_seeker_val_set = read_from_file_with_enter('/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zhang/working/2021_p1_transformer_virome/bai_new/dataset/Seeker/validate&test/PHGVAL2_FF.txt')
    phage_seeker_val_set_ids = list(phage_seeker_val_set.keys())
    phage_seeker_test_set = read_from_file_with_enter('/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zhang/working/2021_p1_transformer_virome/bai_new/dataset/Seeker/validate&test/PHAGE_TEST_FF.txt')
    phage_seeker_test_set_ids = list(phage_seeker_test_set.keys())
    phage_vibrant_set = read_from_file_with_enter('/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zhang/working/2021_p1_transformer_virome/bai_new/dataset/VIBRANT/Dataset/VIBRANT_phage.fasta')
    phage_vibrant_set_ids = list(phage_vibrant_set.keys())
    phage_vibrant_set_ids1 = list(map(split_ids, phage_vibrant_set_ids))
    phage_vibrant_dict = {}
    for i in range(len(phage_vibrant_set_ids1)):
        phage_vibrant_dict[phage_vibrant_set_ids1[i]] = phage_vibrant_set_ids[i]
    banned_list = []
    with open('/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/seeker/train_model/Benchmark_phage.txt') as f:
        for line in f.readlines():
            line = line.strip()
            banned_list.append(line)
    phage_pretraining_list = phage_selfbuilt_ids1 + phage_seeker_training_set1_ids + phage_seeker_training_set2_ids + phage_seeker_val_set_ids + phage_seeker_test_set_ids + phage_vibrant_set_ids1
    phage_pretraining_list = list(set(phage_pretraining_list)) # equal to unique(x) in R
    write_phage_pretrain_fasta(outdir="phage_pretraining_new2.txt", names=phage_pretraining_list, banned_list=banned_list)
    phage_training_list = phage_seeker_training_set1_ids + phage_seeker_training_set2_ids + phage_seeker_val_set_ids + phage_seeker_test_set_ids + phage_vibrant_set_ids1
    phage_training_list = list(set(phage_training_list)) # equal to unique(x) in R
    write_phage_training_fasta(outdir="phage_training_new2.txt", names=phage_training_list, banned_list=banned_list)
    f = open("phage_supplement3.txt", "w")
    for i in range(len(phage_selfbuilt_ids)):
        if (phage_selfbuilt_ids[i].split('.')[0] in banned_list) or (phage_selfbuilt_ids[i].split('.')[0] in phage_vibrant_set_ids1) is True:
            continue
        print('>' + phage_selfbuilt_ids[i], file=f)
        print(phage_selfbuilt[phage_selfbuilt_ids[i]], file=f)
    f.close()
    print("finish all")


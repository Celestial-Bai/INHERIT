#Author: Zeheng Bai
##### INHERIT READ FASTA FILES #####
from basicsetting import *
segment_length = 500
#### This code is inspired by DNABERT ####
def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers

#### These codes are inspired by Seeker ####
def handle_non_ATGC(sequence):
    """
    Handle non ATGCs.
    :param sequence: String input.
    :return: String output (only ATCGs), with randomly assigned bp to non-ATGCs.
    """
    ret = re.sub('[^ATCGN]', 'N', sequence)
    assert len(ret) == len(sequence)
    return ret

def pad_sequence(sequence, source_sequence, length=segment_length):
    """
    Pad sequence by repeating it.
    :param sequence: Segmented sequence to pad.
    :param source_sequence: Original, complete sequence.
    :param length: Length of sequence to pad up to.
    :return: Padded sequence of lenth length.
    """
    assert len(sequence) < length, len(sequence)
    assert sequence == source_sequence or len(source_sequence) > length
    if len(source_sequence) > length:
        ret = source_sequence[len(source_sequence)-len(sequence)-(length-len(sequence)):
                              len(source_sequence)-len(sequence)]+sequence

    else:
        assert sequence == source_sequence
        ret = (source_sequence * (int(length / len(sequence)) + 1))[:length]
    assert len(ret) == length
    return ret


def segment_sequence(sequence, segment_length=segment_length):
    """
    Convert a sequence into list of equally sized segments.
    :param sequence: Sequence of A, C, G, T.
    :param segment_length: Length of segment to divide into.
    :return: List of segments of size segment_length.
    """
    assert len(sequence) > 10, "Sequence is fewer than 200 bases, minimum input is 200 bases"
    ret = []

    for start_idx in range(0, len(sequence), segment_length):
        fragment = sequence[start_idx:(start_idx + segment_length)]
        assert len(fragment) <= segment_length
        if len(fragment) < segment_length:
            fragment = pad_sequence(fragment, sequence, segment_length)
        ret.append(fragment.upper())

    return ret

def segment_fasta(fasta, segment_lengths=segment_length):
    """
    Parse Fasta into segments.
    :param fasta: File handle in Fasta format.
    :param segment_lengths: Length of segments for model.
    :return: Dictionary of Fasta name -> list of segments.
    """

    ret = OrderedDict()
    seq_name = None
    seq_value = None

    count = 0
    for line in fasta:
        line = line.strip()
        if line.startswith(">"):
            # Reached a new sequence in Fasta, if this is not the first sequence let's save the previous one
            random.seed(243789)
            if seq_name is not None:  # This is not the first sequence
                assert seq_value is not None, seq_name

                # Save the sequence to a dictionary
                ret[seq_name] = segment_sequence(seq_value, segment_lengths)
                count += 1
                print("Read Fasta entry {}, total {} entries read".format(seq_name, count), file=sys.stderr)

            seq_name = line.lstrip(">").split()[0]
            seq_value = ""
        else:
            seq_value += handle_non_ATGC(line)

    # Write last entry
    ret[seq_name] = segment_sequence(seq_value)
    count += 1
    print("Read Fasta entry {}, total {} entries read".format(seq_name, count), file=sys.stderr)

    return ret


# revised by Zhang@imsut 20210413
def read_fasta(fasta_path, kmer, fragment_length):
    """Read a Fasta into array of arrays, for DNABERT input."""
    # Read in file
    # fasta = SeqIO.parse(fasta_path, "fasta")

    fasta_dic = segment_fasta(fasta_path, fragment_length)

    matrices = []
    for item_id in fasta_dic:
        for seq in fasta_dic[item_id]:
            seq = seq.upper()

            if re.match('^[ACGTN]+$', str(seq)) is None:
                continue

            matrix = seq2kmer(str(seq), kmer)

            matrices.append(matrix)

    return matrices

def read_from_file_with_enter(filename):
    """
    Read fasta file in a whole sequence way
    :param filename: The fasta file name.
    :return: A dictionary contain the genome sequences and their names.
    """
    fr = open(filename,'r')
    seq = {}
    for line in fr:
        if line.startswith('>'):
            name = line.replace('>', '').split()[0]
            print("Read Fasta entry {}".format(name))
            seq[name] = ''
        else:
            seq[name] += line.replace('\n', '').strip()
    fr.close()
    return seq

def write_fasta_seq_names(seqdict, outdir, banned_list):
    """
    Write sequnece names
    :param seqdict: The format after using read_from_file_with_enter.
    :param banned_list: The names of the test set.
    :return: The list of sequence accessions.
    """
    seq_ids = sorted(list(seqdict.keys()))
    seq_ids1 = list(set(seq_ids))
    assert(len(seq_ids) == len(seq_ids1))
    f = open(outdir, "w")
    for i in range(len(seq_ids)):
        assert (seq_ids[i].split('.')[0] not in banned_list)
        print(seq_ids[i].split('.')[0], file=f)
    f.close()
    print('writing finished')

def write_fasta_cleaned(seqdict, outdir, banned_list):
    seq_ids = list(seqdict.keys())
    f = open(outdir, "w")
    for i in range(len(seq_ids)):
        if (seq_ids[i].split('.')[0] in banned_list) is False:
            print('>' + seq_ids[i], file=f)
            print(seqdict[seq_ids[i]], file=f)
    f.close()
    print("write fasta has been done.")
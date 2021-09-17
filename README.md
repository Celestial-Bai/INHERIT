# IGHERIT

This repository includes the implementation of "Identification of bacteriophages using deep representation model with pre-training". We are still developing this package and we will also try to make some improvements of it, so feel free to report to us if there are any issues occurred.  

Please cite our paper if you want to include or use INHERIT in your research.

# Environment and requirements

We use NVIDIA A100 GPUs to train INHERIT with CUDA 11.2. You should download those packages before you run our codes.
```
##### All #####
argparse
numpy
torch
collections
tqdm

##### transformers #####
##### We will try to use the whole package in the future #####
tokenizers
boto3
filelock
requests
regex
sentencepiece
sacremoses
pydantic
uvicorn
fastapi
starlette
tensorflow
```

# Pre-trained models

The pre-trained models are the important parts of INHERIT. If you want to know which sequences we used on pre-training, you can refer "bac_pretrain_names.txt" and "pha_pretrain_names.txt" to obtain the names of the sequences.

For the checkpoints of the pre-trained models we used, you can find in: 

To pre-train the models, we used DNABERT. However, we used the DNABERT on transformers 4.7.0 (the vanilla one used 2.5.0), but you can still refer those codes to pre-train:

```
cd examples

export KMER=6
export TRAIN_FILE=sample_data/pre/6_3k.txt
export TEST_FILE=sample_data/pre/6_3k.txt
export SOURCE=PATH_TO_DNABERT_REPO
export OUTPUT_PATH=output$KMER

python run_pretrain.py \
    --output_dir $OUTPUT_PATH \
    --model_type=dna \
    --tokenizer_name=dna$KMER \
    --config_name=$SOURCE/src/transformers/dnabert-config/bert-config-$KMER/config.json \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --gradient_accumulation_steps 25 \
    --per_gpu_train_batch_size 10 \
    --per_gpu_eval_batch_size 6 \
    --save_steps 500 \
    --save_total_limit 20 \
    --max_steps 200000 \
    --evaluate_during_training \
    --logging_steps 500 \
    --line_by_line \
    --learning_rate 4e-4 \
    --block_size 512 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.98 \
    --mlm_probability 0.025 \
    --warmup_steps 10000 \
    --overwrite_output_dir \
    --n_process 24
```

referred from: [DNABERT 2.2 Model Training](https://github.com/jerryji1993/DNABERT#2-pre-train-skip-this-section-if-you-fine-tune-on-pre-trained-models)

# Fine-tuning

You can find the names of the sequences we used on "bac_training_names.txt" and "pha_training_names.txt" for training set, and "bac_val_names.txt" and "pha_val_names.txt" for validation set.  Since we have too many hyperparameters, we used "IHT_config.py" to record the default hyperparameters we used. You can change them depending on the different scenario, and you can simply used

```
python3 IHT_training.py
```

to train INHERIT.

# Predict 

You can use our example "test_phage.fasta" as an example to experience how INHERIT predicts them and to know how it performs. You can simply used

```
python3 IGN_predict.py --sequence test_phage.fasta --withpretrain True --model bestacc_IGN_withpre_new.pt --out test_out.txt
```

to have a try.

INHERIT can be downloaded from:


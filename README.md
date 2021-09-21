# INHERIT

This repository includes the implementation of "Identification of bacteriophages using deep representation model with pre-training". We are still developing this package and we will also try to make some improvements of it, so feel free to report to us if there are any issues occurred. INHERIT is a model based on [DNABERT](https://github.com/jerryji1993/DNABERT) , an extension of [Huggingface's Transformers](https://github.com/huggingface/transformers).

Please cite our paper if you want to include or use INHERIT in your research.

## Environment and requirements


We use NVIDIA A100 GPUs to train INHERIT with CUDA 11.2.  We also tested our codes on other GPUs, like V100, and they can run smoothly.

Before you start to use INHERIT, you should download those packages before you run our codes. It should be noted that we will test to use Huggingface's  Transformers directly instead of using the source code in the future.  However, it should be noted that, different from the vanilla DNABERT extending the Transformers on 2.5.0, we used the Transformers 4.7.0 on the **fine-tuning** process (we prove the results are the same). 

```
##### All #####
argparse
numpy
torch
collections
tqdm

##### Transformers #####
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


## Dataset information

We prepared the accessions of the bacterium and phage sequences used on pre-training sets, training sets, validation sets and test sets respectively. You can check them on **Dataset Information.xlsx** in **Supplements.zip**. Not only you can know which sequences we used, but you can also know how we get them.


## Pre-trained models

The pre-trained models are the important parts of INHERIT.  For the checkpoints of the pre-trained models we used, you can find in: [Bacteria pre-trained model download](https://drive.google.com/drive/folders/1zMd5NL69JbnIT3T5eu824bipHddz0Uro?usp=sharing) and [Phage pre-trained model download link](https://drive.google.com/drive/folders/1Cs8SNcG0ryxsAjC-CWGDNTiV4THO-wuu?usp=sharing)

To pre-train the models, we used the original DNABERT codes to pre-train. Therefore, please refer the guides on [DNABERT 2.2 Model Training](https://github.com/jerryji1993/DNABERT#2-pre-train-skip-this-section-if-you-fine-tune-on-pre-trained-models) to get a new pre-trained model if you are interested.  We welcome everyone to build any new pre-trained models to improve the performance of INHERIT. We also post the commands below: 

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


## Fine-tuning

Since we have too many hyperparameters, we used "IHT_config.py" to record the default hyperparameters we used. You can change them depending on the different scenario, and you can simply used:
```
python3 IHT_training.py
```
to train INHERIT.
We also have prepared the code of DNABERT if you want to explore the difference between DNABERT and INHERIT. You can also simply used:

```
python3 DNABERT_training.py
```
to train DNABERT. Both of their training process are straightforward and easy. You do not need to add any other commands.

You can download INHERIT on: [Fine-tuned INHERIT download link](https://drive.google.com/file/d/1uGFZWKoonVMjFHD4bRmutoFMVZRMX6UG/view?usp=sharing)


## Predict 

You can use our example "test_phage.fasta" as an example to experience how INHERIT predicts them and to know how it performs. You can simply used：

```
python3 IGN_predict.py --sequence test_phage.fasta --withpretrain True --model INHERIT.pt --out test_out.txt
```

to have a try.

Here: 

**--withpretrain** means to use DNABERT or INHERIT. If you use INHERIT, you should type **True**, and you should type **False** if you use DNABERT.

**--model** means the directory of the DNABERT or INHERIT file you want to use




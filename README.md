# Identification of bacteriophages using deep representation model with pre-training

![Pipeline_new](https://github.com/Celestial-Bai/INHERIT/blob/master/Figures/Pipeline.png)This repository includes the implementation of "Identification of bacteriophages using deep representation model with pre-training". Our method is developed for identifying phages from the metagenome sequences, and the past methods can be categorized into two types: database-based methods and alignment-free methods.  INHERIT integrated those two types of methods by pre-train-fine-tune paradigm. We compared the proposed method with [VIBRANT](https://github.com/AnantharamanLab/VIBRANT) and [Seeker](https://github.com/gussow/seeker) on a third-party benchmark dataset. Our experiments show that INHERIT achieves better performance than the database-based approach and the alignment-free method, with the best F1-score of 0.9868. 

We are still developing this package and we will also try to make some improvements of it, so feel free to report to us if there are any issues occurred. INHERIT is a model based on [DNABERT](https://github.com/jerryji1993/DNABERT) , an extension of [Huggingface's Transformers](https://github.com/huggingface/transformers).

Our package includes the resources of: 

- The source code of INHERIT
- The pre-trained models and the fine-tuned model (INHERIT)
- The information of the datasets we used
- The benckmark results of Seeker, VIBRANT and INHERIT

Please cite our paper if you want to include or use INHERIT in your research.

## Installation

```
git clone https://github.com/Celestial-Bai/INHERIT.git
cd INHERIT
pip install -r dependencies.txt
```



## Environment and requirements


We use NVIDIA A100 GPUs to train INHERIT with CUDA 11.2.  We also tested our codes on other GPUs, like V100, and they can run smoothly.

Before you start to use INHERIT, you should download those packages before you run our codes. It should be noted that we will test to use Huggingface's  Transformers directly instead of using the source code in the future.  However, it should be noted that, different from the vanilla DNABERT extending the Transformers on 2.5.0, we used the Transformers 4.7.0 on the **fine-tuning** process (we prove the results are the same).  Please confirm your device **cannot have transformers package** when using or training INHERIT.

```
##### All #####
argparse
numpy
torch
tqdm

##### Transformers #####
##### We will try to use the whole package in the future #####
Pillow
black==21.4b0
cookiecutter==1.7.2
dataclasses
datasets
deepspeed>=0.3.16
docutils==0.16.0
fairscale>0.3
faiss-cpu
fastapi
filelock
flake8>=3.8.3
flax>=0.3.2
fugashi>=1.0
huggingface-hub==0.0.8
importlib_metadata
ipadic>=1.0.0,<2.0
isort>=5.5.4
jax>=0.2.8
jieba
keras2onnx
nltk
onnxconverter-common
onnxruntime-tools>=1.4.2
onnxruntime>=1.4.0
packaging
parameterized
protobuf
psutil
pydantic
pytest
pytest-sugar
pytest-xdist
recommonmark
regex!=2019.12.17
requests
rouge-score
sacrebleu>=1.4.12
sacremoses
sagemaker>=2.31.0
scikit-learn==0.24.1
sentencepiece==0.1.91
soundfile
sphinx-copybutton
sphinx-markdown-tables
sphinx-rtd-theme==0.4.3
sphinx==3.2.1
sphinxext-opengraph==0.4.1
starlette
tensorflow-cpu>=2.3
tensorflow>=2.3
timeout-decorator
tokenizers>=0.10.1,<0.11
torchaudio
unidic>=1.0.2
unidic_lite>=1.0.7
uvicorn
```



## Predict 

The pre-trained models are the important parts of INHERIT.  **Please first download those two pre-trained models before you use INHERIT**. For the checkpoints of the pre-trained models we used, you can find in: [Bacteria pre-trained model download link](https://drive.google.com/drive/folders/1zMd5NL69JbnIT3T5eu824bipHddz0Uro?usp=sharing) and [Phage pre-trained model download link](https://drive.google.com/drive/folders/1Cs8SNcG0ryxsAjC-CWGDNTiV4THO-wuu?usp=sharing)

You can use our example "test_phage.fasta" as an example to experience how INHERIT predicts them and to know how it performs. You can simply used：

```
cd INHERIT
python3 IHT_predict.py --sequence test_phage.fasta --withpretrain True --model INHERIT.pt --out test_out.txt
```

to have a try.

Here: 

**--withpretrain** means to use DNABERT or INHERIT. If you use INHERIT, you should type **True**, and you should type **False** if you use DNABERT.

**--model** means the directory of the DNABERT or INHERIT file you want to use

You can download INHERIT on: [Fine-tuned INHERIT download link](https://drive.google.com/file/d/1uGFZWKoonVMjFHD4bRmutoFMVZRMX6UG/view?usp=sharing)

Our output is similar to [Seeker](https://github.com/gussow/seeker), and you can also check the sample results in test_out.txt:

```
    name	category	score
    NC_007636.1	Phage	0.9982988238334656
    NC_030928.1	Phage	0.8466060161590576
    NC_030937.1	Phage	0.8519585132598877
    NC_041844.1	Phage	0.700873851776123
    NC_041845.1	Phage	0.8881644010543823
    NC_041846.1	Phage	0.8881642818450928
    NC_041847.1	Phage	0.8000907301902771
    NC_041848.1	Phage	0.7185130715370178
    NC_041849.1	Phage	0.7798346877098083
    NC_041850.1	Phage	0.897127628326416
```



## Pre-trained models

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
cd INHERIT
python3 IHT_training.py
```

to train INHERIT.
We also have prepared the code of DNABERT if you want to explore the difference between DNABERT and INHERIT. You can also simply used:

```
cd INHERIT
python3 DNABERT_training.py
```

to train DNABERT. Both of their training process are straightforward and easy. You do not need to add any other commands.



## Dataset information and benchmark results

We prepared the accessions of the bacterium and phage sequences used on pre-training sets, training sets, validation sets and test sets respectively. You can check them on **Dataset Information.xlsx** in **Supplements.zip**. Not only you can know which sequences we used, but you can also know how we get them. We also posted our benchmark results of Seeker, VIBRANT and INHERIT. You can check them on **Benchmark Results.xlsx** in **Supplements.zip**. 



## Reference

- [DNABERT](https://github.com/jerryji1993/DNABERT) 
- [Huggingface's Transformers](https://github.com/huggingface/transformers)
- [Seeker](https://github.com/gussow/seeker)
- [VIBRANT](https://github.com/AnantharamanLab/VIBRANT)




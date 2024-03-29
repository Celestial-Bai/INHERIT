# Identification of bacteriophage genome sequences with representation learning

![Pipeline_new](https://github.com/Celestial-Bai/INHERIT/blob/master/pipeline.jpeg)This repository includes the implementation of "Identification of bacteriophage genome sequences with representation learning". Our method is developed for identifying phages from the metagenome sequences, and the past methods can be categorized into two types: database-based methods and alignment-free methods.  INHERIT integrated those two types of methods by pre-train-fine-tune paradigm.  Our experiments show that INHERIT achieves better performance than the database-based approaches and the alignment-free methods.

We are still developing this package and we will also try to make some improvements of it, so feel free to report to us if there are any issues occurred. INHERIT is a model based on [DNABERT](https://github.com/jerryji1993/DNABERT) , an extension of [Huggingface's Transformers](https://github.com/huggingface/transformers). We will still update INHERIT for exploring better performance and more convenient utilization.

Our package includes the resources of: 

- The source code of INHERIT
- The pre-trained models and the fine-tuned model (INHERIT)
 
Please cite our paper if you want to include or use INHERIT in your research: [INHERIT](https://academic.oup.com/bioinformatics/article/38/18/4264/6654586)
 
## Installation

```
git clone https://github.com/Celestial-Bai/INHERIT.git
cd INHERIT
pip install -r dependencies.txt
```



## Environment and requirements


We use NVIDIA A100 GPUs to train INHERIT with CUDA 11.4.  We also tested our codes on other GPUs, like V100, and they can run smoothly.

Before you start to use INHERIT, you should download those packages before you run our codes. We used the Transformers 4.7.0 on the **fine-tuning** process (we prove the results are the same) It should be noted that we will test to use Huggingface's  Transformers directly instead of using the source code in the future for better usage :)

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

The pre-trained models are the important parts of INHERIT.  **Please first download those two pre-trained models before you use INHERIT**. For the checkpoints of the pre-trained models we used, you can find in: [Bacteria pre-trained model download link](https://drive.google.com/drive/folders/1d0ubDne87j5rf5K6DYKSOKxnw_eGV-Dr?usp=sharing) and [Phage pre-trained model download link](https://drive.google.com/drive/folders/17oyt613Hr4984SX7IX72fVP3zPrGADvB?usp=sharing)

You can simply used：

```
cd INHERIT
python3 IHT_predict.py --sequence test.fasta --withpretrain True --model INHERIT.pt --out test_out.txt
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

 
You can download INHERIT on: [Fine-tuned INHERIT download link](https://drive.google.com/file/d/169BaPS4BjK-cmnfabpyjNVTjFtyzhbyC/view?usp=sharing)

## Reference

- [DNABERT](https://github.com/jerryji1993/DNABERT) 
- [Huggingface's Transformers](https://github.com/huggingface/transformers)
- [Seeker](https://github.com/gussow/seeker)
- [VIBRANT](https://github.com/AnantharamanLab/VIBRANT)




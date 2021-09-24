# Author: Zeheng Bai
import setuptools

setuptools.setup(
    name="INHERIT",
    version="1.0",
    author="Zeheng Bai",
    author_email="zeheng@hgc.jp",
    description="BERT model applied for nanopore methyaltion detection",
    long_description="",
    long_description_content_type="",
    url="",
    packages=setuptools.find_packages(),
    package_data={'methBERT':['methBERT/*'],'methBERT.transformers':['methBERT/transformers/*'],'methBERT.transformersegginfo':['methBERT/transformers.egg-info/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['argparse', 'numpy', 'torch', 'collections', 'tqdm', 
    'tokenizer', 'sboto3', 'filelock', 'requests', 'regex', 'sentencepiece', 'sacremoses', 'pydantic', 'uvicorn', 'fastapi', 'starlette', 'tensorflow'],
)

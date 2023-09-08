# SoTaNa: The Open-Source Software Development Assistant

<!-- ![sotana](Figures/sotana.jpg) -->
<div align="center">
<img src=Figures/sotana.jpg width=20% />
</div>

## Environment

```
conda create -n sotana python=3.9 -y
conda activate sotana 
pip install datasets==2.11.0 loralib==0.1.1 sentencepiece==0.1.97 
pip install bitsandbytes==0.37.2 torch==2.0.0 gradio==3.20.1 nltk==3.8.1
pip install prettytable==3.7.0 wandb==0.14.2 fire==0.5.0
pip install openai==0.27.9
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
pip install git+https://github.com/huggingface/transformers.git@fe1f5a639d93c9272856c670cff3b0e1a10d5b2b
```

## Data Generation

```
cd data-generation
bash generation_data.sh
```
The generated data is saved in the `data-generation/output/100000`. 
Due to the limit of uploda size, we split the data into `data_0.json` and `data_1.json`. You can execute `python merge_data.py` to merge them.

## Parameter-Efficient Fine-tuning

```
cd fine-tuning
wandb login
model_size=7
epoch=5
bash fine-tuning.sh ${model_size} ${epoch}
```
The detailed training information is shown in as follows.

| Model    | # llama Param. | # lora Param. | Training Time |
|----------|----------------|---------------|---------------|
| SoTaNa-7B  | 7B             | 8.4M          | 25h35m        |
| SoTaNa-13B | 13B            | 13.1M         | 39h10m        |
| SoTaNa-30B | 30B            | 25.6M         | 48h02m        |


## Inference

### Stack Overflow Question-Answering

#### Obtain the Answering

```
cd inference/stackoverflow-question-answering
model_size=7
bash inference.sh ${model_size}
```

#### Evaluation
```
python evaluation.py --refs_filename xxx --preds_filename xxx 
```

### Code Generation

#### Obtain the Results
```
cd inference/code-generation
model_size=7
bash inference.sh ${model_size}
```

#### Evaluation
```
cd inference/code-generation
python evaluation.py --preds_filename xxx
```

### Code Summarization

#### Obtain the Results

```
cd inference/code-summarization
bash inference.sh ${model_size}
```

#### Evaluation
```
python evaluation.py --refs_filename xxx --preds_filename xxx 
```

 

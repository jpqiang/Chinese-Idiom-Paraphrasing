## Chinese Idiom Paraphrasing

Chinese Idiom Paraphrasing (CIP), which goal is to rephrase the idioms of input sentence to generate a fluent, meaning-preserving sentence without any idiom:

![](paraphrasing.png)

Data in this [dataset](./data) and several approaches: 

- LSTM approach

- Transformer approach

- mt5-seq2seq approach

- mt5-infill approach

- mt5-knowledge approach

### Dependecies

- Python>=3.6
- torch>=1.7.1
- transformers==4.8.0
- fairseq==0.10.2

### Pre-trained model

you can download all pre-trained models [here](https://drive.google.com/file/d/1Kzp5CbocFUJaYeEgvmNiVGOPAAY85BLA/view?usp=sharing), and put it into```model```directory.

If you want train models from scratch, you need uses the pre-trained language models mt5-base ([huggingface](https://huggingface.co/google/mt5-base)) and place the models under the ```model``` directory after downloading.

### Train

train LSTM and Transformer model by fairseq, you need process data for jieba and bpe tokenize sentence, we use scripts from Subword-nmt:

```shell
git clone https://github.com/rsennrich/subword-nmt
```

Then run
```shell
sh prepare.sh
```

train LSTM, Transformer, mt5-seq2seq, mt5-fill, mt5-knowledge model

```shell
sh train_lstm.sh
sh train_transformer.sh
sh train_t5.sh
sh train_t5_fill.sh
sh train_t5_knowledge.sh
```

### Evaluate

Run the following command to evaluate
```shell
sh evaluate_base.sh
sh evaluate_t5.sh
sh evaluate_t5_knowledge.sh
sh evaluate_t5_fill.sh
```

## Citation

```
@article{qiang2022chinese,
  title={Chinese Idiom Paraphrasing},
  author={Qiang, Jipeng and Li, Yang and Zhang, Chaowei and Li, Yun and Yuan, Yunhao and Zhu, Yi and Wu, Xindong},
  journal={arXiv preprint arXiv:2204.07555},
  year={2022}
}
```

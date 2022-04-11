import argparse
import csv
from itertools import islice
import logging
import jieba
import sacrebleu
import torch
from tqdm import tqdm
from transformers import MT5ForConditionalGeneration
from transformers import T5Tokenizer
from bert_score import score

from evaluate_rough import get_rouge_score

logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda')
jieba.initialize()


def load_sys(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip().strip("\n").strip("\t"))
    return data


def contain_idiom(sent, idioms):
    word = set(sent.split())
    contain = list(word & idioms)
    return contain


def tokenize(line, delim=' '):
    _line = line
    _line = _line.replace("\xa0", " ").strip()
    _tok = jieba.cut(_line.rstrip('\r\n'))
    _tokenized = delim.join(_tok)
    return _tokenized


def get_simplify(sent, all_idioms, model, tokenizer, max_length, beam_size, end_token):
    mask_sent = ""
    contain_idioms = contain_idiom(tokenize(sent), all_idioms)

    for idiom in contain_idioms:
        mask_sent = sent.replace(idiom, " <extra_id_0>")
        concat_sent = sent + " [SEP] " + mask_sent
        encoded_sent = tokenizer.encode_plus(concat_sent, add_special_tokens=True, return_tensors='pt')
        input_ids = encoded_sent['input_ids'].to(DEVICE)

        outputs = model.generate(input_ids=input_ids,
                                 num_beams=beam_size,
                                 num_return_sequences=3,
                                 max_length=max_length,
                                 early_stopping=True)

        def _filter(output, _end_token=end_token):
            _txt = tokenizer.decode(output[2:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if _end_token in _txt:
                _end_token_index = _txt.index(_end_token)
                txt = _txt[:_end_token_index]
                return txt
            else:
                return _txt

        results = list(map(_filter, outputs))
        mask_sent = mask_sent.replace(" <extra_id_0>", results[0].replace(" ", "").replace(",", "，"))
    return mask_sent


def read_idioms_list(idiom_path):
    idioms = set()
    with open(idiom_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            idiom = line.strip().strip("\n")
            idioms.add(idiom)
    return idioms


def evaluate(valid_csv, idioms_path, model, tokenizer, max_length, beam_size, end_token):
    predict_sys = []
    predict_ref = []
    all_idioms = read_idioms_list(idioms_path)
    logger.info("***** start evaluating ******")
    with open(valid_csv) as read_file:
        reader = csv.reader(read_file)
        for row in tqdm(islice(reader, 1, None)):
            line_src = row[0].strip(" ").strip("\n")
            line_ref = row[1].strip(" ").strip("\n")
            simplify_result = get_simplify(line_src, all_idioms, model, tokenizer, max_length, beam_size, end_token)
            predict_sys.append(simplify_result)
            predict_ref.append(line_ref)

    bleu_score = sacrebleu.corpus_bleu(predict_sys, [predict_ref], tokenize='zh').score
    rouge_score = get_rouge_score(predict_sys, predict_ref)
    P, R, F = score(predict_sys, predict_ref, lang="zh")

    logging.info(f"  Bleu score : {bleu_score}")
    logging.info(f"  Rouge score : {rouge_score}")
    logging.info(f"  BertScore : P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}")
    logger.info("***** evaluate completes! *****")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='idioms fill')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument('--idioms', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)

    args = parser.parse_args()
    t5_tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    t5_tokenizer.add_special_tokens({'additional_special_tokens': ["[NULL]"]})
    t5_tokenizer.add_special_tokens({'additional_special_tokens': ["[SEP]"]})
    t5_model = MT5ForConditionalGeneration.from_pretrained(args.model_path).to(DEVICE)
    idioms_set = read_idioms_list(args.idioms)
    END_TOKEN = "[NULL]"

    evaluate(args.test_path, idioms_set, t5_model, t5_tokenizer, args.max_length, args.beam, END_TOKEN)

import argparse
import csv
from itertools import islice
import logging
import jieba
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration
from train_infill import T5PegasusTokenizer

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

        outputs = model.generate(
            input_ids=input_ids,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_length=max_length,
            early_stopping=True
        )

        def _filter(output, _end_token=end_token):
            _txt = tokenizer.decode(output[2:], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if _end_token in _txt:
                _end_token_index = _txt.index(_end_token)
                txt = _txt[len(" <extra_id_0>"):_end_token_index]
                return txt
            else:
                return _txt

        results = list(map(_filter, outputs))
        mask_sent = mask_sent.replace(" <extra_id_0>", results[0].replace(" ", ""))

    return mask_sent


def read_idioms_list(idiom_path):
    idioms = set()
    with open(idiom_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            idiom = line.strip().strip("\n")
            idioms.add(idiom)
    return idioms


def evaluate(valid_csv, output_path, idioms_set, model, tokenizer, max_length, beam_size, end_token):
    predict_sys = []
    predict_ref = []
    logger.info("***** start evaluating ******")
    with open(valid_csv) as read_file:
        reader = csv.reader(read_file)
        for row in tqdm(islice(reader, 1, None)):
            line_src = row[0].strip(" ").strip("\n").replace(" ", "")
            line_ref = row[1].strip(" ").strip("\n").replace(" ", "")
            simplify_result = get_simplify(
                line_src, idioms_set, model, tokenizer, max_length, beam_size, end_token
            )
            predict_sys.append(simplify_result)
            predict_ref.append(line_ref)

    write_gen_file(predict_sys, output_path)


def write_gen_file(predict_sys, out_file):
    with open(out_file, 'a+') as f:
        for line in predict_sys:
            f.write(line + '\n')
    print('write complete!')


def main():
    parser = argparse.ArgumentParser(description='idioms fill')
    parser.add_argument('--model_path', type=str, default="model/t5_pagesus_fill_in/7")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument('--idioms', type=str, default="idioms.txt")
    parser.add_argument('--test_path', type=str, default="data/in_domain/test.in.csv")
    parser.add_argument('--output_path', type=str, default="result/fill_in/test.t5.in.fill.hpy")

    args = parser.parse_args()
    t5_tokenizer = T5PegasusTokenizer.from_pretrained(args.model_path)
    t5_tokenizer.add_tokens("<extra_id_0>")
    t5_tokenizer.add_tokens("<extra_id_1>")
    t5_tokenizer.add_tokens("[null]")
    t5_model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(DEVICE)
    idioms_set = read_idioms_list(args.idioms)
    end_token = "[null]"

    evaluate(
        args.test_path, args.output_path, idioms_set, t5_model, t5_tokenizer, args.max_length, args.beam, end_token
    )


if __name__ == '__main__':
    main()

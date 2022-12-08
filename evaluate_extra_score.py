import argparse
import jieba
import sacrebleu
from rouge import Rouge
from bert_score import scorer

rouge = Rouge()
jieba.initialize()
bert_scorer = scorer.BERTScorer(lang="zh", device="cuda")


def read_idioms_set(idiom_path):
    idioms = set()
    with open(idiom_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            idiom = line.strip().strip("\n")
            idioms.add(idiom)
    return idioms


def contain_idiom(sent, idioms):
    sent = tokenize(sent.replace(" ", ""), "word")
    word = set(sent.split())
    contain = list(word & idioms)
    return contain


# Word segmentation
def tokenize(line, tokenized_method, delim=' '):
    line = line.replace("\xa0", " ").strip()
    if tokenized_method == "word":
        tok = jieba.cut(line.rstrip('\r\n'))
    elif tokenized_method == "char":
        tok = [c for c in line.rstrip('\r\n')]
    else:
        raise ValueError("Need either a tokenized method use 'word' or 'char'.")

    return delim.join(tok)


def preprocess(src_file, hpy_file, ref_file, idioms, tokenized_method="word"):
    """
    tokenized_method: tokenize sentence use jieba for word or space for chinese char
    """
    all_source = []
    all_hypothesis = []
    all_references = []
    with open(src_file, 'r', encoding='utf-8') as f_src:
        for lines_src in f_src.readlines():
            line_src = lines_src.replace(" ", "")
            all_source.append(line_src)

    with open(hpy_file, 'r', encoding='utf-8') as f_hpy:
        for lines_hpy in f_hpy.readlines():
            line_hpy = lines_hpy.replace(" ", "")
            all_hypothesis.append(line_hpy)

    with open(ref_file, 'r', encoding='utf-8') as f_ref:
        for lines_ref in f_ref.readlines():
            line_ref = lines_ref.replace(" ", "")
            all_references.append(line_ref)

    remove_hypothesis = []
    remove_references = []
    for src, hpy, ref in zip(all_source, all_hypothesis, all_references):
        hpy = tokenize(hpy, tokenized_method)
        ref = tokenize(ref, tokenized_method)

        contain_idioms = contain_idiom(src, idioms)
        src = tokenize(src, tokenized_method) if tokenized_method == "word" else src
        no_idiom_src = ""
        for _idiom in contain_idioms:
            no_idiom_src = src.replace(_idiom + " ", "") if tokenized_method == "word" else src.replace(_idiom, "")

        no_idiom_src = no_idiom_src if tokenized_method == "word" else tokenize(no_idiom_src, tokenized_method)

        match_token = list(set(no_idiom_src.split(" ")) & set(hpy.split(" ")) & set(ref.split(" ")))
        while len(match_token) >= 1:
            hpy = remove_match(hpy, match_token)
            ref = remove_match(ref, match_token)

            match_token = list(set(no_idiom_src.split(" ")) & set(hpy.split(" ")) & set(ref.split(" ")))

        remove_hypothesis.append(hpy.replace(" ", ""))
        remove_references.append(ref.replace(" ", ""))

    return remove_hypothesis, remove_references


def remove_match(sent, remove_list):
    sent_word = sent.split(" ")
    new_word_list = sent_word

    # only remove match token in sentence one time
    for remove_token in remove_list:
        if remove_token in sent_word:
            new_word_list.remove(remove_token)

    return " ".join(new_word_list)


def compute_raw_score(sys, ref):
    hypothesis = []
    references = []

    with open(sys, 'r', encoding='utf-8') as f_sys:
        for lines_sys in f_sys:
            line_sys = lines_sys.strip().replace(" ", "")
            hypothesis.append(line_sys)

    with open(ref, 'r', encoding='utf-8') as f_ref:
        for lines_ref in f_ref:
            line_ref = lines_ref.strip().replace(" ", "")
            references.append(line_ref)

    assert len(hypothesis) == len(references)

    all_scores = compute_all_score(hypothesis, references)

    return all_scores


def compute_all_score(preds, labels):
    hpy, ref = postprocess_text(preds, labels)
    assert len(hpy) == len(ref)

    rough_scores = rouge.get_scores(hpy, ref, avg=True)
    for key in rough_scores:
        rough_scores[key] = rough_scores[key]['f'] * 100

    hpy = [i.replace(" ", "") for i in hpy]
    ref = [i.replace(" ", "") for i in ref]
    bleu_score = sacrebleu.corpus_bleu(hpy, [ref], tokenize='zh').score

    P, R, F1 = bert_scorer.score(hpy, ref)

    return rough_scores, bleu_score, F1


def postprocess_text(preds, labels):
    preds = [tokenize(item.replace(" ", ""), "char") for item in preds]
    labels = [tokenize(item.replace(" ", ""), "char") for item in labels]

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    while '.' in preds:
        idx = preds.index('.')
        preds[idx] = '。'

    while '' in preds:
        idx = preds.index('')
        preds[idx] = '。'

    while '' in labels:
        idx = labels.index('')
        labels[idx] = '。'

    assert len(preds) == len(labels)
    return preds, labels


def main():
    parser = argparse.ArgumentParser(description='input and output file path')
    parser.add_argument('--idioms', type=str, default="idioms.txt")
    parser.add_argument('--src', type=str, default="data/in_domain/test.in.src")
    parser.add_argument('--hyp', type=str,
                        default="result/lstm_in/result_beam_5_81.61/generate_beam_5.sys")
    parser.add_argument('--ref', type=str,
                        default="data/in_domain/test.in.dst")
    args = parser.parse_args()

    idioms_set = read_idioms_set(args.idioms)

    extra_hypothesis, extra_references = preprocess(
        args.src, args.hyp, args.ref,
        idioms_set,
        tokenized_method="word",
    )

    extra_rouge, extra_bleu, extra_bert = compute_all_score(extra_hypothesis, extra_references)
    print("extra_rouge: ", extra_rouge)
    print("extra_bleu: ", extra_bleu)
    print("extra_bert: ", extra_bert.mean().item())
    print(" ")

    raw_rough, raw_bleu, raw_bert = compute_raw_score(args.hyp, args.ref)
    print("raw_rough: ", raw_rough)
    print("raw_bleu: ", raw_bleu)
    print("raw_bert: ", raw_bert.mean().item())


if __name__ == '__main__':
    main()

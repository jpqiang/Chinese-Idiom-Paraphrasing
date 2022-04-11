import argparse
import rouge_zh


# Word segmentation
def tokenize(line, delim=' '):
    line = line.replace("\xa0", " ").strip()
    tok = [c for c in line.rstrip('\r\n')]
    tokenized = delim.join(tok)
    return tokenized


def get_file(sys_file, ref_file):
    all_hypothesis = []
    all_references = []
    with open(sys_file, 'r', encoding='utf-8') as f0:
        for lines0 in f0.readlines():
            line0 = lines0.strip()
            line0 = tokenize(line0)
            all_hypothesis.append(line0)

    with open(ref_file, 'r', encoding='utf-8') as f1:
        for lines1 in f1.readlines():
            line1 = lines1.strip()
            line1 = [tokenize(line1)]
            all_references.append(line1)

    return all_hypothesis, all_references


def get_rouge_score(sys, ref):
    hypothesis = []
    references = []

    for lines0 in sys:
        line0 = lines0.strip()
        line0 = tokenize(line0)
        hypothesis.append(line0)

    for lines1 in ref:
        line1 = lines1.strip()
        line1 = [tokenize(line1)]
        references.append(line1)

    evaluator = rouge_zh.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                               max_n=4,
                               limit_length=True,
                               length_limit=256,
                               length_limit_type='words',
                               alpha=0.5,  # Default F1_score
                               weight_factor=1.2,
                               stemming=True)

    rouge_scores = evaluator.get_scores(hypothesis, references)

    return rouge_scores


def main():
    parser = argparse.ArgumentParser(description='input and output file path')
    parser.add_argument('--sys', type=str, default=None)
    parser.add_argument('--ref', type=str, default=None)
    args = parser.parse_args()

    hypothesis, references = get_file(args.sys, args.ref)
    evaluator = rouge_zh.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                               max_n=4,
                               limit_length=True,
                               length_limit=256,
                               length_limit_type='words',
                               alpha=0.5,  # Default F1_score
                               weight_factor=1.2,
                               stemming=True)

    rouge_scores = evaluator.get_scores(hypothesis, references)

    print(rouge_scores)


if __name__ == '__main__':
    main()

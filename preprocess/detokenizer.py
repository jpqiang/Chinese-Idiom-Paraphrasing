import argparse

parser = argparse.ArgumentParser(description='input and output file path')
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()


def detoken(input_path, output_path):
    detok = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for lines in f.readlines():
            line = lines.strip().replace(' ', '')
            detok.append(line)

    st = '\n'
    f = open(output_path, "w", encoding='utf-8')
    f.write(st.join(detok))
    f.close()
    print("detokenize completed!")


if __name__ == "__main__":
    detoken(args.input, args.output)

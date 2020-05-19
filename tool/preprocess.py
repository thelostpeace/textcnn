from tokenizer import Tokenizer
import argparse

def parse_data(data, word_tokenize):
    info = data.split('\t')
    token = word_tokenize(info[1])

    return "%s\t%s" % (info[0], ' '.join(token))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tokenize data")
    parser.add_argument('--input', type=str, required=True, help='input file')
    parser.add_argument('--output', type=str, required=True, help='output file')
    args = parser.parse_args()

    with open(args.input) as rf:
        with open(args.output, "w+") as wf:
            tokenizer = Tokenizer().tokenize
            for line in rf:
                out = parse_data(line.strip(), tokenizer)
                wf.write("%s\n" % out)

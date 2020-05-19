import os
os.environ["NLTK_DATA"] = "/home/work/liushaohua/software/nltk_data"
from nltk.tokenize import word_tokenize

class Tokenizer:
    '''
        对于中文字符，两边加空格，tokenize用nltk的tokenizer
    '''
    def __init__(self):
        '''
            ascii_valid will stay in sequence
            ascii_invalid will be erased and treated as delimeter
            we list all cjk except cjk compatibility and cjk radicals/kangxi radicals
            for chinese tokenization we just use Han and cjk punctuation
        '''
        self.tokenizer = word_tokenize
        self.han_begin = "\u4e00"
        self.han_end = "\u9fff"
        self.cjk_ea_begin = "\u3400"
        self.cjk_ea_end = "\u4dbf"
        self.cjk_eb_begin = "\u0002\u0000"
        self.cjk_eb_end = "\u0002\ua6df"
        self.cjk_ec_begin = "\u0002\ua700"
        self.cjk_ec_end = "\u0002\ub73f"
        self.cjk_ed_begin = "\u0002\ub740"
        self.cjk_ed_end = "\u0002\ub81f"
        self.cjk_ee_begin = "\u0002\ub820"
        self.cjk_ee_end = "\u0002\uceaf"
        self.cjk_ef_begin = "\u0002\uceb0"
        self.cjk_ef_end = "\u0002\uebe0"
        self.cjk_eg_begin = "\u0003\u0000"
        self.cjk_eg_end = "\u0003\u134a"
        self.cjk_punc_begin = "\u3000"
        self.cjk_punc_end = "\u303f"
        self.cjk_punc_comp_begin = "\ufe30"
        self.cjk_punc_comp_end = "\ufe4f"
        self.cjk_symbol_begin = "\u3200"
        self.cjk_symbol_end = "\u32ff"
        self.cjk_symbol_comp_begin = "\u3300"
        self.cjk_symbol_comp_end = "\u33ff"

    def tokenize(self, s):
        seq = ""
        for c in s:
            if (c >= self.han_begin and c <= self.han_end) or (c >= self.cjk_punc_begin and c <= self.cjk_punc_end):
                seq += " " + c + " "
            else:
                seq += c

        return self.tokenizer(seq)

'''
if __name__ == '__main__':
    tokenizer = Tokenizer()
    s = "今天A股上升了1.2%。"
    print(tokenizer.tokenize(s))
'''

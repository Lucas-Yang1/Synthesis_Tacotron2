import re


__pad = '_'
__eos = '~'
__unk = '<unk>'
_chars = 'abcdefghijklmnopqrstuvwxyz1234567890!?;:., '

symbols = [__pad, __eos, __unk] + list(_chars)

symbol2idx = {s:i for i,s in enumerate(symbols)}
idx2symbol = {i:s for i,s in enumerate(symbols)}

_whitespace_re = re.compile(r"\s+")

def lowercase(text: str):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

def tokenizer(text: str):
    token = [symbol2idx.get(s, symbol2idx[__unk]) for s in collapse_whitespace(lowercase(text))]
    token.append(symbol2idx[__eos])
    return token

def batch_tokenizer(text: list[str]):
    max_len = max(map(len, text))
    token = [None] * len(text)
    text_len = [None] * len(text)
    for i in range(len(text)):
        token[i] = tokenizer(text[i]) + [0] * (max_len-len(text[i]))
        text_len[i] = len(text[i])
    return token, text_len
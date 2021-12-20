import re


__pad = '_'
__eos = '~'
_chars = 'abcdefghijklmnopqrstuvwxyz1234567890!?;:., '

symbols = [__pad, __eos] + list(_chars)

symbol2idx = {s:i for i,s in enumerate(symbols)}
idx2symbol = {i:s for i,s in enumerate(symbols)}

_whitespace_re = re.compile(r"\s+")

def lowercase(text: str):
    return text.lower()

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

def tokenizer(text: str):

     return [symbol2idx[s] for s in collapse_whitespace(lowercase(text))]
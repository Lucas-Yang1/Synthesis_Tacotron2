from pypinyin import lazy_pinyin, Style
def procecess_transcript(fpath):
    data_info = {}
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            lines = line.strip().split()
            filename = lines[0]
            words = ' '.join(lines[1:])
            data_info[filename] = words

    return data_info

if __name__ == '__main__':
    fpath = 'D:/dataset/aidatatang_200zh/aidatatang_200zh~/aidatatang_200zh/transcript/aidatatang_200_zh_transcript.txt'
    data_info= procecess_transcript(fpath)
from pathlib import Path

from tqdm import tqdm

from preprocess_speaker import preprocess_speaker_general
from multiprocessing.pool import Pool
from preprocess_transcript import procecess_transcript
from functools import partial
dataset_info = {
    "aidatatang_200zh": {
        "subfolders": ["corpus/train"],
        "trans_filepath": "transcript/aidatatang_200_zh_transcript.txt",
        "speak_func": preprocess_speaker_general
    },
}


def process_dataset(dataset_root: Path, out_dir: Path, n_processes: int,
                    skip_existing: bool, dataset: str):
    # create dir
    out_dir.joinpath(dataset, 'mel').mkdir(parents=True, exist_ok=True)
    out_dir.joinpath(dataset, 'audio').mkdir(parents=True, exist_ok=True)
    out_dir.joinpath(dataset, 'embedding').mkdir(parents=True, exist_ok=True)

    #
    func = dataset_info[dataset]["speak_func"]
    speaker_dirs = [i for i in dataset_root.joinpath(dataset_info[dataset]['subfolders'][0]).glob('*') if i.is_dir()]

    data_info = procecess_transcript(dataset_root.joinpath(dataset_info[dataset]['trans_filepath']))
    func = partial(func, out_dir=out_dir.joinpath(dataset), skip_existing=skip_existing, data_info=data_info)
    job = Pool(n_processes).imap(func, speaker_dirs)

    metafpath = out_dir.joinpath(dataset,'_metafile.txt')
    metafile = open(metafpath, 'a' if skip_existing else 'w')

    for metadatas in tqdm(job, dataset, len(speaker_dirs), unit='speaker'):
        for metadata in metadatas:
            metafile.write("|".join(str(s) for s in metadata) + '\n')
    metafile.close()



def embed_utterance():
    pass

if __name__ == '__main__':
    dataset_root = Path('D:/dataset/aidatatang_200zh/')
    out_dir = Path('./postdata')
    process_dataset(dataset_root, out_dir, 2, False, "aidatatang_200zh")
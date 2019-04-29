"""
This file generates tokenization, POS features, stemmed words from a spacy service.
"""
import json
import pdb
import glob
from util.spacy.spacy_client import SpacyClient
client = SpacyClient()
req_batch_size = 200

def nli_jsonl_reader(fpath):
    for l in open(fpath, 'r', encoding='utf8'):
        o = json.loads(l.strip())
        yield o['sentence1'], o['sentence2'], {k:o[k] for k in o if ('sentence' not in k and 'annotator' not in k)}

def annotate_corpus(fplst):
    """
    The input is a list of files of paraphrase pairs.
    The input format should be tab separated cols: label, '-', '-', sentence1, sentence2
    The result will be written to file named with same prefix and different suffix.
    :return:
    """
    for fp in fplst:
        outputfp = fp[:fp.rfind('.')]+'.spacy.jsonl'
        req_batch=[]
        with open(outputfp, 'w', encoding='utf8') as outputf:
            for s1, s2, otherdict in nli_jsonl_reader(fp):
                req_batch.append(s1)
                req_batch.append(s2)
                if len(req_batch) == req_batch_size:
                    resp = client.annotate_input(req_batch)
                    assert len(resp)==req_batch_size
                    for i in range(len(resp)//2):
                        s1ann = json.loads(resp[i*2])
                        s2ann = json.loads(resp[i*2+1])
                        o = {'sentence1':s1ann, 'sentence2':s2ann}
                        o = {**o, **otherdict}
                        outputf.writelines(json.dumps(o)+'\n')
                    req_batch=[]
        print("Done annotating file", fp)

#annotate_corpus(glob.glob('/Users/huiyingli/Documents/DIIN/data/multinli_0.9/*.jsonl'))
annotate_corpus(glob.glob('/Users/huiyingli/Documents/DIIN/data/snli_1.0/*.jsonl'))
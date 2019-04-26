"""
This file generates tokenization, POS features, stemmed words from a spacy service.
"""
import json
import pdb
from util.spacy.spacy_client import SpacyClient
client = SpacyClient()
req_batch_size = 200

def nli_jsonl_reader(fpath):
    for l in open(fpath, 'r', encoding='utf8'):
        o = json.loads(l.strip())
        sent1 = o['sentence1']
        sent2 = o['sentence2']
        yield  sent1, sent2, o['captionID'], o['pairID'],o['gold_label']

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
            for s1, s2, captionID, pairID, gold_label in nli_jsonl_reader(fp):
                req_batch.append(s1)
                req_batch.append(s2)
                if len(req_batch) == req_batch_size:
                    resp = client.annotate_input(req_batch)
                    assert len(resp)==req_batch_size
                    for i in range(len(resp)//2):
                        s1ann = json.loads(resp[i*2])
                        s2ann = json.loads(resp[i*2+1])
                        o = {'sentence1':s1ann, 'sentence2':s2ann, 'pairID':pairID, 'captionID':captionID, 'gold_label':gold_label}
                        outputf.writelines(json.dumps(o)+'\n')
                    req_batch=[]

annotate_corpus(['/Users/huiyingli/Documents/DIIN/data/snli_1.0/snli_1.0_{}.jsonl'.format(data) for data in ['train','dev','test']])
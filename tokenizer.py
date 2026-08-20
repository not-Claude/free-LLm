import re
from collections import Counter

class CharTokenizer:
    def __init__(self, max_vocab=4096):
        self.max_vocab=max_vocab
        self.stoi={}
        self.itos=[]

    def build(self, text):
        counts=Counter(text)
        chars=[c for c,_ in counts.most_common(self.max_vocab-4)]
        self.itos=["<pad>","<unk>","<bos>","<eos>"]+chars
        self.stoi={c:i for i,c in enumerate(self.itos)}

    def encode(self,text):
        u=self.stoi.get
        unk=self.stoi["<unk>"]
        return [u(c,unk) for c in text]

    def decode(self,ids):
        return "".join(self.itos[i] for i in ids if i < len(self.itos))

    @property
    def vocab_size(self):
        return len(self.itos)

    def save(self,path):
        with open(path,"w",encoding="utf8") as f:
            for x in self.itos:
                f.write(repr(x)+"\n")

    @classmethod
    def load(cls,path):
        t=cls()
        with open(path,encoding="utf8") as f:
            t.itos=[eval(x.rstrip("\n")) for x in f]
        t.stoi={c:i for i,c in enumerate(t.itos)}
        return t

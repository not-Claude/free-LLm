import argparse, os, gc, time
import torch
from model import TinyTransformer
from tokenizer import CharTokenizer

def read_sample(path, limit=8_000_000):
    with open(path,"r",encoding="utf8",errors="ignore") as f:
        return f.read(limit)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--steps",type=int,default=100000)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--block-size",type=int,default=256)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--checkpoint",default="checkpoint.pt")
    args=p.parse_args()

    os.makedirs("checkpoints",exist_ok=True)

    sample=read_sample(args.data)
    tok=CharTokenizer(4096)
    tok.build(sample)
    tok.save("tokenizer.txt")
    del sample
    gc.collect()

    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))

    model=TinyTransformer(tok.vocab_size,block_size=args.block_size).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)

    if os.path.exists(args.checkpoint):
        state=torch.load(args.checkpoint,map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start=state["step"]+1
        print("Продолжение с шага",start)
    else:
        start=0

    with open(args.data,"r",encoding="utf8",errors="ignore") as f:
        text=f.read()

    ids=torch.tensor(tok.encode(text),dtype=torch.long)
    del text
    gc.collect()

    # Для настоящего большого корпуса заменяйте это на mmap/чанки.
    # Этот демонстрационный загрузчик намеренно ограничивает RAM.
    if ids.numel() > 20_000_000:
        ids=ids[:20_000_000]

    model.train()
    for step in range(start,args.steps):
        ix=torch.randint(0,ids.numel()-args.block_size-1,(args.batch_size,))
        x=torch.stack([ids[i:i+args.block_size] for i in ix]).to(device)
        y=torch.stack([ids[i+1:i+args.block_size+1] for i in ix]).to(device)

        logits=model(x)
        loss=torch.nn.functional.cross_entropy(
            logits.reshape(-1,tok.vocab_size),y.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()

        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.4f}")
        if step % 1000 == 0 and step:
            torch.save({
                "step":step,
                "model":model.state_dict(),
                "optimizer":opt.state_dict()
            },args.checkpoint)

if __name__=="__main__":
    main()

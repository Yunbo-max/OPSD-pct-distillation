#!/usr/bin/env python3
"""Sequential confidence-calibrated NVIP audit on screened frontier rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pct.nvip import sequential_audit_decision

try:
    from .extract_predictable_field_dataset import deployment_prompt
    from .run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt
except ImportError:
    from extract_predictable_field_dataset import deployment_prompt
    from run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt


def seed_for(seed: int, identity: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{identity}".encode()).digest()[:8], "big") % (2**31-1)


def logits(model, ids):
    with torch.inference_mode():
        return model(ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits[0, -1].float()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("frontier_cache"); ap.add_argument("--out",required=True)
    ap.add_argument("--model",default="Qwen/Qwen3-4B"); ap.add_argument("--n_nodes",type=int,default=24)
    ap.add_argument("--top_k",type=int,default=4); ap.add_argument("--initial_b",type=int,default=4)
    ap.add_argument("--max_b",type=int,default=16); ap.add_argument("--max_new_tokens",type=int,default=512)
    ap.add_argument("--node_stride",type=int,default=8); ap.add_argument("--delta",type=float,default=.01)
    ap.add_argument("--seed",type=int,default=20260826)
    args=ap.parse_args()
    rows={str(x["id"]):x for x in map(json.loads,Path(args.dataset).open())}
    caches=[]
    for f in sorted(Path(args.frontier_cache).glob("*.pt")):
        x=torch.load(f,map_location="cpu",weights_only=False)
        if 0<int(x["successes"])<8: caches.append((f,x))
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map="cuda").eval()
    output=Path(args.out); output.parent.mkdir(parents=True,exist_ok=True); output.touch(exist_ok=True)
    done={json.loads(s)["id"] for s in output.read_text().splitlines() if s}
    written=0
    with output.open("a") as handle:
      for f,cache in caches:
        if written>=args.n_nodes: break
        row=rows[str(cache["id"])]
        # Pick a successful rollout when available, but select the node only by JS(p,q).
        ridx=int(torch.where(cache["rewards"]>0)[0][0]) if int(cache["successes"]) else 0
        completion=cache["completion_ids"][ridx]
        valid=(completion!=tok.pad_token_id).nonzero().flatten()
        positions=valid[8:-8:args.node_stride]
        if not len(positions): continue
        sp=tok(deployment_prompt(tok,row["problem"]),return_tensors="pt",add_special_tokens=False).input_ids.cuda()
        tp=tok(online_teacher_prompt(tok,row["problem"],row["references"][0],True),return_tensors="pt",add_special_tokens=False).input_ids.cuda()
        best=None
        for pos_t in positions:
            pos=int(pos_t); pref=completion[:pos].unsqueeze(0).cuda()
            pl=logits(model,torch.cat((sp,pref),1)); ql=logits(model,torch.cat((tp,pref),1))
            pp=pl.softmax(0); qq=ql.softmax(0); mm=(pp+qq)/2
            js=.5*((pp*(pp.clamp_min(1e-30).log()-mm.log())).sum()+(qq*(qq.clamp_min(1e-30).log()-mm.log())).sum())
            if best is None or float(js)>best[0]: best=(float(js),pos,pl.cpu(),ql.cpu())
        js,pos,pl,ql=best; key=f'{cache["id"]}|{ridx}|{pos}'
        if key in done: continue
        pfull=pl.softmax(0); qfull=ql.softmax(0)
        ids=torch.unique(torch.cat((pfull.topk(args.top_k).indices,qfull.topk(args.top_k).indices)))
        mask=torch.ones_like(pfull,dtype=torch.bool); mask[ids]=False
        # OTHER is a fixed tail policy mu ∝ p+q; projection may change only its total mass.
        mu=(pfull+qfull)*mask; mu=mu/mu.sum()
        p=torch.cat((pfull[ids],pfull[mask].sum().view(1)))
        q=torch.cat((qfull[ids],qfull[mask].sum().view(1)))
        successes=torch.zeros(len(ids)+1,dtype=torch.long); trials=torch.zeros_like(successes)
        prefix=completion[:pos]
        def sample(branch,b):
            candidate=int(ids[branch]) if branch<len(ids) else int(torch.multinomial(mu,1).item())
            node=torch.cat((sp.cpu()[0],prefix,torch.tensor([candidate]))).unsqueeze(0).cuda()
            torch.manual_seed(seed_for(args.seed,f"{key}|{branch}|{b}"))
            with torch.inference_mode():
                out=model.generate(node,attention_mask=torch.ones_like(node),do_sample=True,temperature=.7,top_p=.95,
                    max_new_tokens=args.max_new_tokens,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
            text=tok.decode(torch.cat((prefix,torch.tensor([candidate]),out[0,node.shape[1]:].cpu())),skip_special_tokens=True)
            return int(answers_equivalent(boxed(text),row["answer"]))
        for branch in range(len(trials)):
            for b in range(args.initial_b): successes[branch]+=sample(branch,b)
            trials[branch]=args.initial_b
        while True:
            audit=sequential_audit_decision(p,q,successes,trials,delta=args.delta,max_trials=args.max_b)
            if audit["next_action"] is None: break
            branch=int(audit["next_action"]); b=int(trials[branch]); successes[branch]+=sample(branch,b); trials[branch]+=1
        record={"id":key,"problem_id":str(cache["id"]),"rollout":ridx,"position":pos,"js":js,
            "candidate_ids":ids.tolist(),"has_other":True,"student_probs":p.tolist(),"teacher_probs":q.tolist(),
            "successes":successes.tolist(),"trials":trials.tolist(),"decision":audit["decision"],
            "advantage_estimate":float(audit["estimate"]),"advantage_lower":float(audit["lower"]),
            "advantage_upper":float(audit["upper"]),"total_rollouts":audit["total_rollouts"]}
        handle.write(json.dumps(record)+"\n"); handle.flush(); written+=1
        print(f'node={written} {key} js={js:.4f} decision={audit["decision"]} CI=[{float(audit["lower"]):+.4f},{float(audit["upper"]):+.4f}] B={audit["total_rollouts"]}',flush=True)

if __name__=="__main__": main()

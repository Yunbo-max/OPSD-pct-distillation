#!/usr/bin/env python3
"""Stage-A signed teacher-advantage audit via Jordan decomposition."""
from __future__ import annotations
import argparse, hashlib, json, math, random, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pct.nvip import wilson_interval
try:
 from .extract_predictable_field_dataset import deployment_prompt
 from .run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt
except ImportError:
 from extract_predictable_field_dataset import deployment_prompt
 from run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt

def seed_for(seed, identity):
 return int.from_bytes(hashlib.sha256(f'{seed}|{identity}'.encode()).digest()[:8], 'big')%(2**31-1)
def next_logits(model, ids):
 with torch.inference_mode(): return model(ids,attention_mask=torch.ones_like(ids),logits_to_keep=1).logits[0,-1].float()
def eb_radius(x, confidence=.95):
 # Empirical-Bernstein radius for a variable in [-1,1].
 n=x.numel(); log=math.log(3.0/(1-confidence)); var=x.var(unbiased=False)
 return float(torch.sqrt(2*var*log/n)+6*log/n)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('frontier_cache'); ap.add_argument('--out',required=True)
 ap.add_argument('--model',default='Qwen/Qwen3-4B'); ap.add_argument('--n_nodes',type=int,default=100)
 ap.add_argument('--nodes_per_problem',type=int,default=2); ap.add_argument('--initial_pairs',type=int,default=8)
 ap.add_argument('--max_pairs',type=int,default=64); ap.add_argument('--delta',type=float,default=.01)
 ap.add_argument('--max_new_tokens',type=int,default=512); ap.add_argument('--seed',type=int,default=20260827)
 ap.add_argument('--pair_batch_size',type=int,default=8)
 args=ap.parse_args(); rows={str(x['id']):x for x in map(json.loads,Path(args.dataset).open())}
 tok=AutoTokenizer.from_pretrained(args.model)
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval()
 # First rank candidate nodes by TV, without looking at any branch outcomes.
 ranked=[]
 for f in sorted(Path(args.frontier_cache).glob('*.pt')):
  c=torch.load(f,map_location='cpu',weights_only=False)
  if not 0<int(c['successes'])<8: continue
  row=rows[str(c['id'])]; ridx=int(torch.where(c['rewards']>0)[0][0]); completion=c['completion_ids'][ridx]
  valid=(completion!=tok.pad_token_id).nonzero().flatten(); positions=valid[8:-8:8]
  if not len(positions): continue
  sp=tok(deployment_prompt(tok,row['problem']),return_tensors='pt',add_special_tokens=False).input_ids.cuda()
  tp=tok(online_teacher_prompt(tok,row['problem'],row['references'][0],True),return_tensors='pt',add_special_tokens=False).input_ids.cuda(); best=None
  for pt in positions:
   pos=int(pt); pref=completion[:pos].unsqueeze(0).cuda(); p=next_logits(model,torch.cat((sp,pref),1)).softmax(0); q=next_logits(model,torch.cat((tp,pref),1)).softmax(0)
   tv=.5*(p-q).abs().sum()
   if best is None or float(tv)>best[0]: best=(float(tv),pos)
  ranked.append((best[0],str(c['id']),ridx,best[1],f,completion))
 ranked.sort(reverse=True); selected=[]
 counts={}
 for item in ranked:
  if len(selected)>=args.n_nodes: break
  if counts.get(item[1],0)>=args.nodes_per_problem: continue
  counts[item[1]]=counts.get(item[1],0)+1; selected.append(item)
 out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.touch(exist_ok=True)
 done={json.loads(s)['id'] for s in out.read_text().splitlines() if s}
 with out.open('a') as h:
  for ni,(tv,pid,ridx,pos,f,completion) in enumerate(selected):
   key=f'{pid}|{ridx}|{pos}'
   if key in done: continue
   row=rows[pid]; sp=tok(deployment_prompt(tok,row['problem']),return_tensors='pt',add_special_tokens=False).input_ids.cuda(); tp=tok(online_teacher_prompt(tok,row['problem'],row['references'][0],True),return_tensors='pt',add_special_tokens=False).input_ids.cuda(); pref=completion[:pos]
   p=next_logits(model,torch.cat((sp,pref.unsqueeze(0).cuda()),1)).softmax(0); q=next_logits(model,torch.cat((tp,pref.unsqueeze(0).cuda()),1)).softmax(0)
   diff=q-p; alpha=float(diff.clamp_min(0).sum()); nuplus=diff.clamp_min(0)/max(alpha,1e-30); numinus=(-diff).clamp_min(0)/max(alpha,1e-30)
   ds=[]; target=args.initial_pairs
   def sample_pairs(n_pairs, start):
    plus=torch.multinomial(nuplus,n_pairs,replacement=True)
    minus=torch.multinomial(numinus,n_pairs,replacement=True)
    actions=torch.cat((plus,minus))
    common=torch.cat((sp[0].cpu(),pref)).unsqueeze(0).expand(2*n_pairs,-1)
    nodes=torch.cat((common,actions.unsqueeze(1)),1).cuda()
    torch.manual_seed(seed_for(args.seed,f'{key}|batch|{start}'))
    with torch.inference_mode():
     generated=model.generate(nodes,attention_mask=torch.ones_like(nodes),do_sample=True,temperature=.7,top_p=.95,
       max_new_tokens=args.max_new_tokens,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    out=[]
    for j in range(2*n_pairs):
     text=tok.decode(torch.cat((pref,actions[j:j+1],generated[j,nodes.shape[1]:].cpu())),skip_special_tokens=True)
     out.append(int(answers_equivalent(boxed(text),row['answer'])))
    return [out[i]-out[n_pairs+i] for i in range(n_pairs)]
   while True:
    while len(ds)<target:
     n_pairs=min(args.pair_batch_size,target-len(ds)); ds.extend(sample_pairs(n_pairs,len(ds)))
    x=torch.tensor(ds,dtype=torch.float32); mean=float(x.mean()); radius=eb_radius(x)
    # D is bounded in [-1,1], hence |A_T| <= alpha=TV(p,q).
    lower=alpha*max(-1.0, mean-radius); upper=alpha*min(1.0, mean+radius)
    if lower>args.delta: decision='helpful'; break
    if upper<-args.delta: decision='harmful'; break
    if target>=args.max_pairs: decision='abstain'; break
    target=min(args.max_pairs,target*2)
   rec={'id':key,'problem_id':pid,'rollout':ridx,'position':pos,'tv':tv,'alpha':alpha,'pairs':len(ds),'signed_samples':ds,'advantage_estimate':alpha*mean,'lower':lower,'upper':upper,'decision':decision}
   h.write(json.dumps(rec)+'\n'); h.flush(); print(f'node={ni+1} {key} TV={tv:.3f} alpha={alpha:.3f} {decision} CI=[{lower:+.4f},{upper:+.4f}] pairs={len(ds)}',flush=True)
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Fixed-node terminal/control-variate benchmark with batched paired rollouts."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pct.advantage_estimators import (
 doubly_robust_advantage, jordan_signed_measures, predict_ridge_value, signed_terminal_advantage,
)
try:
 from .extract_predictable_field_dataset import deployment_prompt
 from .run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt
except ImportError:
 from extract_predictable_field_dataset import deployment_prompt
 from run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt

def seed_for(seed,identity):
 return int.from_bytes(hashlib.sha256(f'{seed}|{identity}'.encode()).digest()[:8],'big')%(2**31-1)
def next_logits(model,ids):
 with torch.inference_mode(): return model(ids,attention_mask=torch.ones_like(ids),logits_to_keep=1).logits[0,-1].float()
def interval_dict(x): return {k:getattr(x,k) for k in ('estimate','lower','upper','radius','samples')}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('frontier_cache'); ap.add_argument('nodes'); ap.add_argument('value_head')
 ap.add_argument('--out',required=True); ap.add_argument('--model',default='Qwen/Qwen3-4B'); ap.add_argument('--n_nodes',type=int,default=26)
 ap.add_argument('--proxy_pairs',type=int,default=32); ap.add_argument('--residual_pairs',type=int,default=8)
 ap.add_argument('--short_horizon',type=int,default=32); ap.add_argument('--max_new_tokens',type=int,default=512)
 ap.add_argument('--batch_pairs',type=int,default=4); ap.add_argument('--seed',type=int,default=20260829); ap.add_argument('--confidence',type=float,default=.95)
 ap.add_argument('--coupling',choices=('common','independent'),default='common')
 args=ap.parse_args(); rows={str(x['id']):x for x in map(json.loads,Path(args.dataset).open())}
 node_rows=[json.loads(s) for s in Path(args.nodes).read_text().splitlines() if s][:args.n_nodes]
 cache={}
 for f in Path(args.frontier_cache).glob('*.pt'):
  x=torch.load(f,map_location='cpu',weights_only=False); cache[str(x['id'])]=x
 head=torch.load(args.value_head,map_location='cpu',weights_only=False); mean=head['feature_mean']; std=head['feature_std']; weights=head['weights']
 tok=AutoTokenizer.from_pretrained(args.model)
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval()
 out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.touch(exist_ok=True); done={json.loads(s)['id'] for s in out.read_text().splitlines() if s}
 def value(sequences):
  mask=torch.ones_like(sequences)
  with torch.inference_mode(): result=model(sequences,attention_mask=mask,output_hidden_states=True,return_dict=True)
  feature=result.hidden_states[-1][:,-1].float().cpu(); return predict_ridge_value((feature-mean)/std,weights)
 def short_pair_batch(common,nuplus,numinus,n,key,offset):
  torch.manual_seed(seed_for(args.seed,f'{key}|actions|{offset}')); plus=torch.multinomial(nuplus,n,replacement=True); minus=torch.multinomial(numinus,n,replacement=True)
  def arm(actions,side):
   nodes=torch.cat((common.expand(n,-1),actions.cpu().unsqueeze(1)),1).cuda()
   shared=seed_for(args.seed,f'{key}|short|{offset}')
   # Resetting the seed gives plus/minus arms common random numbers.
   torch.manual_seed(shared if args.coupling=='common' else seed_for(shared,side))
   with torch.inference_mode(): generated=model.generate(nodes,attention_mask=torch.ones_like(nodes),do_sample=True,temperature=.7,top_p=.95,min_new_tokens=args.short_horizon,max_new_tokens=args.short_horizon,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
   return generated,value(generated)
  gp,vp=arm(plus,'plus'); gm,vm=arm(minus,'minus'); return plus,minus,gp,gm,vp,vm
 with out.open('a') as handle:
  for index,node in enumerate(node_rows):
   key=node['id']
   if key in done: continue
   pid=str(node['problem_id']); stored=cache[pid]; row=rows[pid]; ridx=int(node['rollout']); pos=int(node['position']); prefix=stored['completion_ids'][ridx,:pos]
   sp=tok(deployment_prompt(tok,row['problem']),return_tensors='pt',add_special_tokens=False).input_ids.cuda(); tp=tok(online_teacher_prompt(tok,row['problem'],row['references'][0],True),return_tensors='pt',add_special_tokens=False).input_ids.cuda(); pref=prefix.unsqueeze(0).cuda()
   p=next_logits(model,torch.cat((sp,pref),1)).softmax(0); q=next_logits(model,torch.cat((tp,pref),1)).softmax(0); alpha,nuplus,numinus=jordan_signed_measures(p,q)
   common=torch.cat((sp.cpu(),prefix.unsqueeze(0)),1); proxy=[]; residual=[]; terminal_plus=[]; terminal_minus=[]
   for begin in range(0,args.proxy_pairs,args.batch_pairs):
    n=min(args.batch_pairs,args.proxy_pairs-begin); _,_,_,_,vp,vm=short_pair_batch(common,nuplus,numinus,n,key,begin); proxy.extend((vp-vm).tolist())
   for begin in range(0,args.residual_pairs,args.batch_pairs):
    n=min(args.batch_pairs,args.residual_pairs-begin); plus,minus,gp,gm,vp,vm=short_pair_batch(common,nuplus,numinus,n,key,100000+begin)
    def finish(generated,side):
     base_seed=seed_for(args.seed,f'{key}|terminal|{begin}')
     torch.manual_seed(base_seed if args.coupling=='common' else seed_for(base_seed,side))
     with torch.inference_mode(): full=model.generate(generated,attention_mask=torch.ones_like(generated),do_sample=True,temperature=.7,top_p=.95,max_new_tokens=max(1,args.max_new_tokens-args.short_horizon),pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
     return [int(answers_equivalent(boxed(tok.decode(seq[sp.shape[1]:],skip_special_tokens=True)),row['answer'])) for seq in full.cpu()]
    rp=finish(gp,'plus'); rm=finish(gm,'minus'); terminal_plus.extend(rp); terminal_minus.extend(rm)
    residual.extend(((torch.tensor(rp)-torch.tensor(rm))-(vp-vm)).tolist())
   terminal=signed_terminal_advantage(torch.tensor(terminal_plus),torch.tensor(terminal_minus),alpha,confidence=args.confidence)
   dr=doubly_robust_advantage(torch.tensor(proxy),torch.tensor(residual),alpha,confidence=args.confidence)
   record={'id':key,'problem_id':pid,'tv':float(alpha),'proxy_differences':proxy,'audited_residuals':residual,
    'terminal_plus':terminal_plus,'terminal_minus':terminal_minus,'terminal':interval_dict(terminal),'doubly_robust':interval_dict(dr),
    'config':{'proxy_pairs':args.proxy_pairs,'residual_pairs':args.residual_pairs,'short_horizon':args.short_horizon,'max_new_tokens':args.max_new_tokens,'coupling':args.coupling}}
   handle.write(json.dumps(record)+'\n'); handle.flush(); print(f'node={index+1} {key} terminal_width={terminal.upper-terminal.lower:.3f} dr_width={dr.upper-dr.lower:.3f}',flush=True)
if __name__=='__main__': main()

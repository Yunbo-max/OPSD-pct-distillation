#!/usr/bin/env python3
"""Extract frozen-model prefix features for a cross-fitted success value head."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
 from .extract_predictable_field_dataset import deployment_prompt
except ImportError:
 from extract_predictable_field_dataset import deployment_prompt

def excluded_problem_ids(path: str | None) -> set[str]:
 if not path: return set()
 return {str(json.loads(line)["problem_id"]) for line in Path(path).read_text().splitlines() if line}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('frontier_cache'); ap.add_argument('--out',required=True)
 ap.add_argument('--model',default='Qwen/Qwen3-4B'); ap.add_argument('--fractions',default='.15,.35,.55,.75,.9')
 ap.add_argument('--batch_size',type=int,default=4); ap.add_argument('--exclude_nodes'); ap.add_argument('--n_problems',type=int,default=500)
 args=ap.parse_args(); fractions=[float(x) for x in args.fractions.split(',')]
 rows={str(x['id']):x for x in map(json.loads,Path(args.dataset).open())}; excluded=excluded_problem_ids(args.exclude_nodes)
 tok=AutoTokenizer.from_pretrained(args.model)
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval()
 examples=[]
 for path in sorted(Path(args.frontier_cache).glob('*.pt'))[:args.n_problems]:
  c=torch.load(path,map_location='cpu',weights_only=False); pid=str(c['id'])
  if pid in excluded: continue
  prompt=tok(deployment_prompt(tok,rows[pid]['problem']),return_tensors='pt',add_special_tokens=False).input_ids[0]
  for ridx,completion in enumerate(c['completion_ids']):
   valid=int((completion!=tok.pad_token_id).sum())
   for fraction in fractions:
    length=max(1,min(valid,int(valid*fraction)))
    examples.append((torch.cat((prompt,completion[:length])),float(c['rewards'][ridx]),pid,ridx,fraction))
 features=[]; outcomes=[]; problem_ids=[]; metadata=[]
 for begin in range(0,len(examples),args.batch_size):
  batch=examples[begin:begin+args.batch_size]; width=max(x[0].numel() for x in batch)
  ids=torch.full((len(batch),width),tok.pad_token_id,dtype=torch.long); mask=torch.zeros_like(ids)
  for i,(seq,*_) in enumerate(batch): ids[i,:seq.numel()]=seq; mask[i,:seq.numel()]=1
  with torch.inference_mode():
   output=model(ids.cuda(),attention_mask=mask.cuda(),output_hidden_states=True,return_dict=True)
  last=output.hidden_states[-1]; indices=mask.sum(1).cuda()-1
  feature=last[torch.arange(len(batch),device=last.device),indices].float().cpu()
  features.append(feature)
  for _,reward,pid,ridx,fraction in batch:
   outcomes.append(reward); problem_ids.append(pid); metadata.append({'rollout':ridx,'fraction':fraction})
  if begin % (args.batch_size*25)==0: print(f'extracted={begin+len(batch)}/{len(examples)}',flush=True)
 destination=Path(args.out); destination.parent.mkdir(parents=True,exist_ok=True)
 torch.save({'features':torch.cat(features),'outcomes':torch.tensor(outcomes),'problem_ids':problem_ids,
             'metadata':metadata,'model':args.model,'fractions':fractions,'excluded_problem_ids':sorted(excluded)},destination)
 print(f'wrote {destination} examples={len(outcomes)} hidden_dim={features[0].shape[1]}',flush=True)
if __name__=='__main__': main()

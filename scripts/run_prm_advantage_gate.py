#!/usr/bin/env python3
"""PRM next-step gate on the frozen Jordan development nodes.

Phase 1 uses Qwen3-4B to sample plus/minus Jordan branches and cache both a
short next-step text and an independent terminal answer.  Phase 2 unloads the
student and scores the short step with Qwen2.5-Math-PRM-7B.  The models are
never resident together, which keeps this gate within a 24GB GPU.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pct.advantage_estimators import jordan_signed_measures
try:
 from .extract_predictable_field_dataset import deployment_prompt
 from .run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt
except ImportError:
 from extract_predictable_field_dataset import deployment_prompt
 from run_online_paired_residual import answers_equivalent, boxed, online_teacher_prompt

def seed_for(seed, identity):
 return int.from_bytes(hashlib.sha256(f'{seed}|{identity}'.encode()).digest()[:8],'big')%(2**31-1)
def next_logits(model,ids):
 with torch.inference_mode(): return model(ids,attention_mask=torch.ones_like(ids),logits_to_keep=1).logits[0,-1].float()
def step_text(text, cap=64):
 # Keep exactly one complete next step when a separator appears; otherwise cap
 # by characters as a conservative fallback for a malformed generation.
 pieces=re.split(r'\n\s*\n',text, maxsplit=1)
 return pieces[0][:cap*4] if len(pieces)==1 else pieces[0]
def prm_score(model,tokenizer,problem,response):
 system='Please reason step by step, and put your final answer within \\boxed{}.'
 steps=[x.strip() for x in re.split(r'\n\s*\n',response) if x.strip()]
 if not steps: steps=[response.strip()]
 response_with_markers='<extra_0>'.join(steps)+'<extra_0>'
 messages=[{'role':'system','content':system},{'role':'user','content':problem},{'role':'assistant','content':response_with_markers}]
 text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
 ids=tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids.to(model.device)
 sep=tokenizer.encode('<extra_0>',add_special_tokens=False)[0]
 # The released remote code currently expects the legacy cache API.  Scoring
 # a complete step does not need KV reuse, so disable cache for compatibility.
 with torch.inference_mode(): output=model(input_ids=ids,use_cache=False)
 positions=(ids[0]==sep).nonzero().flatten()
 if not len(positions): return float('nan')
 probs=torch.softmax(output.logits[0,positions].float(),dim=-1)
 return float(probs[-1,1])

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('frontier_cache'); ap.add_argument('nodes'); ap.add_argument('--out',required=True)
 ap.add_argument('--student_model',default='Qwen/Qwen3-4B'); ap.add_argument('--prm_model',default='Qwen/Qwen2.5-Math-PRM-7B')
 ap.add_argument('--n_nodes',type=int,default=5); ap.add_argument('--pairs',type=int,default=8); ap.add_argument('--batch_pairs',type=int,default=4)
 ap.add_argument('--short_tokens',type=int,default=64); ap.add_argument('--terminal_tokens',type=int,default=512); ap.add_argument('--seed',type=int,default=20260830)
 ap.add_argument('--phase',choices=('all','generate','score'),default='all'); args=ap.parse_args()
 rows={str(x['id']):x for x in map(json.loads,Path(args.dataset).open())}; node_rows=[json.loads(s) for s in Path(args.nodes).read_text().splitlines() if s][:args.n_nodes]
 cache={}
 for f in Path(args.frontier_cache).glob('*.pt'):
  x=torch.load(f,map_location='cpu',weights_only=False); cache[str(x['id'])]=x
 out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
 generated_path=out.with_suffix('.branches.jsonl')
 if args.phase in ('all','generate'):
  tok=AutoTokenizer.from_pretrained(args.student_model)
  if tok.pad_token_id is None: tok.pad_token=tok.eos_token
  student=AutoModelForCausalLM.from_pretrained(args.student_model,dtype=torch.bfloat16,device_map='cuda').eval()
  with generated_path.open('w') as handle:
   for ni,node in enumerate(node_rows):
    pid=str(node['problem_id']); row=rows[pid]; stored=cache[pid]; ridx=int(node['rollout']); pos=int(node['position']); prefix=stored['completion_ids'][ridx,:pos]
    prompt=tok(deployment_prompt(tok,row['problem']),return_tensors='pt',add_special_tokens=False).input_ids.cuda(); teacher=tok(online_teacher_prompt(tok,row['problem'],row['references'][0],True),return_tensors='pt',add_special_tokens=False).input_ids.cuda(); pref=prefix.unsqueeze(0).cuda()
    p=next_logits(student,torch.cat((prompt,pref),1)).softmax(0); q=next_logits(student,torch.cat((teacher,pref),1)).softmax(0); alpha,nuplus,numinus=jordan_signed_measures(p,q)
    prefix_text=tok.decode(prefix,skip_special_tokens=True); common=torch.cat((prompt.cpu(),prefix.unsqueeze(0)),1); records=[]
    for begin in range(0,args.pairs,args.batch_pairs):
     n=min(args.batch_pairs,args.pairs-begin); torch.manual_seed(seed_for(args.seed,f'{pid}|actions|{begin}')); plus=torch.multinomial(nuplus,n,replacement=True); minus=torch.multinomial(numinus,n,replacement=True); actions=torch.cat((plus.cpu(),minus.cpu()))
     nodes=torch.cat((common.expand(2*n,-1),actions.unsqueeze(1)),1).cuda(); torch.manual_seed(seed_for(args.seed,f'{pid}|short|{begin}'))
     with torch.inference_mode(): short=student.generate(nodes,attention_mask=torch.ones_like(nodes),do_sample=True,temperature=.7,top_p=.95,min_new_tokens=args.short_tokens,max_new_tokens=args.short_tokens,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
     torch.manual_seed(seed_for(args.seed,f'{pid}|terminal|{begin}'))
     with torch.inference_mode(): full=student.generate(short,attention_mask=torch.ones_like(short),do_sample=True,temperature=.7,top_p=.95,max_new_tokens=max(1,args.terminal_tokens-args.short_tokens),pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
     for j in range(n):
      # Decode only the newly generated suffix; the prefix is supplied
      # separately to the PRM and must not determine the step boundary.
      suffix_start=prompt.shape[1]+prefix.numel()+1
      plus_text=tok.decode(short[j,suffix_start:],skip_special_tokens=True); minus_text=tok.decode(short[n+j,suffix_start:],skip_special_tokens=True)
      plus_full=tok.decode(full[j,prompt.shape[1]:],skip_special_tokens=True); minus_full=tok.decode(full[n+j,prompt.shape[1]:],skip_special_tokens=True)
      records.append({'plus_action':tok.decode([int(plus[j])],skip_special_tokens=True),'minus_action':tok.decode([int(minus[j])],skip_special_tokens=True),
       'plus_step':step_text(plus_text),'minus_step':step_text(minus_text),'plus_terminal':int(answers_equivalent(boxed(plus_full),row['answer'])),'minus_terminal':int(answers_equivalent(boxed(minus_full),row['answer']))})
    handle.write(json.dumps({'id':f'{pid}|{ridx}|{pos}','problem_id':pid,'position':pos,'alpha':float(alpha),'prefix_text':prefix_text,'records':records})+'\n'); handle.flush(); print(f'generated node={ni+1}/{len(node_rows)} {pid}',flush=True)
  del student; del tok; torch.cuda.empty_cache()
 if args.phase in ('all','score'):
  prm_tok=AutoTokenizer.from_pretrained(args.prm_model,trust_remote_code=True); prm=AutoModel.from_pretrained(args.prm_model,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='auto').eval()
  result_rows=[]
  for line in generated_path.read_text().splitlines():
   row=json.loads(line); problem=rows[row['problem_id']]['problem']; plus=[]; minus=[]; terminal=[]
   for record in row['records']:
    plus.append(prm_score(prm,prm_tok,problem,row['prefix_text']+' '+record['plus_action']+record['plus_step']))
    minus.append(prm_score(prm,prm_tok,problem,row['prefix_text']+' '+record['minus_action']+record['minus_step']))
    terminal.append(record['plus_terminal']-record['minus_terminal'])
   row['prm_plus']=plus; row['prm_minus']=minus; row['prm_differences']=[a-b for a,b in zip(plus,minus)]; row['terminal_differences']=terminal; result_rows.append(row); print(f'scored {row["id"]}',flush=True)
  out.write_text(json.dumps({'model':args.prm_model,'nodes':result_rows},indent=2)+'\n')
if __name__=='__main__': main()

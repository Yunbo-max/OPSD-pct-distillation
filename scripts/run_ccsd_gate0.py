#!/usr/bin/env python3
"""Gate 0: test whether a student correctness gradient has free-generation value."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_privileged_effect_diagnostic import student_prompt
from scripts.run_online_paired_residual import answers_equivalent, boxed

def seed_for(seed,identity):
 return int.from_bytes(hashlib.sha256(f'{seed}|{identity}'.encode()).digest()[:8],'big')%(2**31-1)
def parse_layers(value): return [int(x) for x in value.split(',') if x.strip()]
def pad_batch(sequences,pad):
 width=max(x.numel() for x in sequences); ids=torch.full((len(sequences),width),pad,dtype=torch.long)
 for i,x in enumerate(sequences): ids[i,:x.numel()]=x
 return ids

def contrastive_gradients(model, correct_ids, incorrect_ids, correct_len, incorrect_len, layers, prefix_len):
 def branch_gradient(ids,step_len):
  captured={}
  def hook(layer):
   def capture(_module,_inputs,output):
    hidden=output[0] if isinstance(output,tuple) else output; hidden.retain_grad(); captured[layer]=hidden; return output
   return capture
  handles=[model.model.layers[layer].register_forward_hook(hook(layer)) for layer in layers]
  try:
   model.zero_grad(set_to_none=True)
   with torch.enable_grad():
    # Keep only the logits needed for the step. The final extra position is
    # discarded because it predicts the token after the labeled continuation.
    output=model(input_ids=ids.unsqueeze(0),attention_mask=torch.ones(1,ids.numel(),device=model.device,dtype=torch.long),
      use_cache=False,logits_to_keep=step_len+1,return_dict=True)
    logits=output.logits.float()[0,:step_len]; target=ids[prefix_len:prefix_len+step_len]
    score=F.log_softmax(logits,dim=-1).gather(-1,target.unsqueeze(-1)).squeeze(-1).mean(); score.backward()
    result={}
    for layer in layers:
     hidden=captured[layer]; state=hidden[0,prefix_len-1].detach().float(); gradient=hidden.grad[0,prefix_len-1].detach().float()
     result[layer]=(gradient,state)
    return result,float(score.detach())
  finally:
   for handle in handles: handle.remove()
   model.zero_grad(set_to_none=True); del captured
 correct,correct_score=branch_gradient(correct_ids.to(model.device),correct_len)
 incorrect,incorrect_score=branch_gradient(incorrect_ids.to(model.device),incorrect_len)
 result={}
 for layer in layers:
  correct_grad,state=correct[layer]; incorrect_grad,_=incorrect[layer]; gradient=correct_grad-incorrect_grad
  rms=state.square().mean().sqrt().clamp_min(1e-6); grad_rms=gradient.square().mean().sqrt().clamp_min(1e-6)
  result[layer]=(gradient,state,float(grad_rms),float(rms))
 return result,correct_score,incorrect_score

def generate_arm(model,tokenizer,prefix_ids,delta,seed,max_new_tokens):
 torch.manual_seed(seed)
 state={'patched':False}
 handle=None
 if delta is not None:
  addition=delta.to(device=model.device,dtype=next(model.parameters()).dtype)
  def hook(_module,_inputs,output):
   hidden=output[0] if isinstance(output,tuple) else output
   if state['patched']: return output
   modified=hidden.clone(); modified[:,-1,:]+=addition; state['patched']=True
   return (modified,*output[1:]) if isinstance(output,tuple) else modified
  handle=model.model.layers[delta.layer].register_forward_hook(hook)
 try:
  with torch.inference_mode(): generated=model.generate(prefix_ids,attention_mask=torch.ones_like(prefix_ids),do_sample=True,temperature=.7,top_p=.95,max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
  return generated[0,prefix_ids.shape[1]:].cpu()
 finally:
  if handle is not None: handle.remove()

def generate_arm_batch(model,tokenizer,prefix_ids,deltas,seed,max_new_tokens):
 """Generate fixed-size base/+gradient/-gradient/random arms in one call.

 EOS is disabled so that generation never shrinks the batch before the
 one-time, row-aligned intervention is applied. This is substantially faster
 than four serial calls and is used only for the Gate-0 pilot.
 """
 batch=prefix_ids.repeat(len(deltas),1)
 template=next((d.tensor for d in deltas if d is not None),None)
 if template is None:
  raise ValueError("at least one non-null delta is required")
 values=torch.stack([torch.zeros_like(template) if d is None else d.tensor for d in deltas]).to(model.device)
 state={'patched':False}
 layer=next(d.layer for d in deltas if d is not None)
 def hook(_module,_inputs,output):
  hidden=output[0] if isinstance(output,tuple) else output
  if state['patched']: return output
  modified=hidden.clone(); modified[:,-1,:]+=values.to(dtype=modified.dtype)
  state['patched']=True
  return (modified,*output[1:]) if isinstance(output,tuple) else modified
 handle=model.model.layers[layer].register_forward_hook(hook)
 try:
  torch.manual_seed(seed)
  with torch.inference_mode():
   generated=model.generate(batch,attention_mask=torch.ones_like(batch),do_sample=True,
    temperature=.7,top_p=.95,max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id,
    eos_token_id=None,use_cache=True)
  return generated[:,prefix_ids.shape[1]:].cpu()
 finally:
  handle.remove()

class LayerDelta:
 def __init__(self,layer,tensor): self.layer=layer; self.tensor=tensor
 def to(self,**kwargs): return self.tensor.to(**kwargs)

def generate_arms(model,tokenizer,prefix_ids,deltas,seed,max_new_tokens):
 """Generate base/+g/-g/random as one batch with a one-time layer patch."""
 batch=prefix_ids.expand(len(deltas),-1); additions={}
 for layer,values in deltas.items():
  template=next(value.tensor for value in values if value is not None)
  additions[layer]=torch.stack([torch.zeros_like(template) if value is None else value.tensor for value in values]).to(model.device)
 state={'patched':False}; handles=[]
 for layer,values in additions.items():
  def make_hook(layer,values):
   def hook(_module,_inputs,output):
    hidden=output[0] if isinstance(output,tuple) else output
    if state['patched']: return output
    modified=hidden.clone(); modified[:,-1,:]+=values.to(dtype=modified.dtype); state['patched']=True; return (modified,*output[1:]) if isinstance(output,tuple) else modified
   return hook
  handles.append((layer,model.model.layers[layer].register_forward_hook(make_hook(layer,values))))
 try:
  torch.manual_seed(seed)
  # Keep all four arms in the batch; early EOS would shrink the batch and
  # invalidate row-aligned intervention vectors.
  with torch.inference_mode(): generated=model.generate(batch,attention_mask=torch.ones_like(batch),do_sample=True,temperature=.7,top_p=.95,max_new_tokens=max_new_tokens,pad_token_id=tokenizer.pad_token_id,eos_token_id=None,use_cache=False)
  state['patched']=True
  return generated[:,prefix_ids.shape[1]:].cpu()
 finally:
  for _,handle in handles: handle.remove()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--out',required=True); ap.add_argument('--model',default='Qwen/Qwen3-4B')
 ap.add_argument('--n_examples',type=int,default=20); ap.add_argument('--layers',default='13,17,21,23,25'); ap.add_argument('--alphas',default='.025,.05,.1,.2')
 ap.add_argument('--max_new_tokens',type=int,default=256); ap.add_argument('--seed',type=int,default=20260831); ap.add_argument('--shard_count',type=int,default=1); ap.add_argument('--shard_index',type=int,default=0)
 args=ap.parse_args(); records=[json.loads(s) for s in Path(args.dataset).read_text().splitlines() if s][:args.n_examples]; records=[x for i,x in enumerate(records) if i%args.shard_count==args.shard_index]
 tok=AutoTokenizer.from_pretrained(args.model)
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda').eval(); layers=parse_layers(args.layers); alphas=[float(x) for x in args.alphas.split(',')]
 model.config.pad_token_id=tok.pad_token_id
 out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); completed=sum(1 for s in out.read_text().splitlines() if s) if out.exists() else 0; started=time.time()
 with out.open('a') as handle:
  for index,row in enumerate(records[completed:],start=completed):
   prompt=tok(student_prompt(tok,row['problem']),return_tensors='pt',add_special_tokens=False).input_ids[0]
   prefix=tok(row['paired_prefix'],return_tensors='pt',add_special_tokens=False).input_ids[0]
   correct=tok(row['correct_step'],return_tensors='pt',add_special_tokens=False).input_ids[0]; incorrect=tok(row['incorrect_step'],return_tensors='pt',add_special_tokens=False).input_ids[0]
   prefix_len=prompt.numel()+prefix.numel(); correct_ids=torch.cat((prompt,prefix,correct)); incorrect_ids=torch.cat((prompt,prefix,incorrect)); base_ids=torch.cat((prompt,prefix)).unsqueeze(0).to(model.device)
   result={'id':row.get('id',index),'problem_id':row.get('source_sample_index',index),'layers':{}}
   gradients,correct_score,incorrect_score=contrastive_gradients(model,correct_ids.to(model.device),incorrect_ids.to(model.device),correct.numel(),incorrect.numel(),layers,prefix_len)
   for layer in layers:
    gradient,state,grad_rms,state_rms=gradients[layer]; norm_matched=F.normalize(gradient,dim=0)*state.norm().clamp_min(1e-6)
    bucket=result['layers'].setdefault(str(layer),{'gradient_rms':grad_rms,'state_rms':state_rms,'local_margin':correct_score-incorrect_score,'arms':{}})
    for alpha in alphas:
     delta=LayerDelta(layer,gradient/(gradient.square().mean().sqrt().clamp_min(1e-6))*state.square().mean().sqrt().clamp_min(1e-6)*alpha)
     random=LayerDelta(layer,norm_matched*alpha)
     arm_specs=(('base',None),('plus_gradient',delta),('minus_gradient',LayerDelta(layer,-delta.tensor)),('random',random))
     suffixes=generate_arm_batch(model,tok,base_ids,[change for _,change in arm_specs],
      seed_for(args.seed,f'{row.get("id",index)}|{layer}|{alpha}'),args.max_new_tokens)
     for (arm,_),suffix in zip(arm_specs,suffixes):
      text=tok.decode(suffix,skip_special_tokens=True); reward=float(answers_equivalent(boxed(text),row['answer']))
      bucket['arms'][f'alpha={alpha:g}:{arm}']={'reward':reward,'boxed':boxed(text)}
   handle.write(json.dumps(result,ensure_ascii=False)+'\n'); handle.flush(); print(f'row={index+1}/{len(records)} elapsed={time.time()-started:.1f}s',flush=True)
   del gradients, correct_ids, incorrect_ids, base_ids, correct, incorrect, prefix, prompt
   torch.cuda.empty_cache()
if __name__=='__main__': main()

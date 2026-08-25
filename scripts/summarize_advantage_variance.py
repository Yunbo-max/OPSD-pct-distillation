#!/usr/bin/env python3
"""Summarize terminal versus doubly robust advantage precision and cost."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('inputs',nargs='+'); ap.add_argument('--out')
 args=ap.parse_args(); rows=[]
 for name in args.inputs:
  rows.extend(json.loads(s) for s in Path(name).read_text().splitlines() if s)
 summaries=[]
 for row in rows:
  terminal=row['terminal']; dr=row['doubly_robust']; config=row['config']
  signed=torch.tensor(row['terminal_plus']).float()-torch.tensor(row['terminal_minus']).float()
  residual=torch.tensor(row['audited_residuals']).float()
  terminal_variance=float(signed.var(unbiased=False)) if len(signed)>1 else float('nan')
  residual_variance=float(residual.var(unbiased=False)) if len(residual)>1 else float('nan')
  terminal_width=terminal['upper']-terminal['lower']; dr_width=dr['upper']-dr['lower']
  terminal_tokens=2*len(signed)*config['max_new_tokens']
  dr_tokens=2*(len(row['proxy_differences'])*config['short_horizon']+len(residual)*config['max_new_tokens'])
  summaries.append({'id':row['id'],'terminal_width':terminal_width,'dr_width':dr_width,
   'width_ratio':dr_width/max(terminal_width,1e-12),'terminal_variance':terminal_variance,
   'residual_variance':residual_variance,'variance_ratio':residual_variance/max(terminal_variance,1e-12),
   'terminal_tokens':terminal_tokens,'dr_tokens':dr_tokens,'token_ratio':dr_tokens/max(terminal_tokens,1)})
 def mean(field): return sum(x[field] for x in summaries)/max(len(summaries),1)
 aggregate={'nodes':len(summaries),'mean_terminal_width':mean('terminal_width'),'mean_dr_width':mean('dr_width'),
  'mean_width_ratio':mean('width_ratio'),'mean_variance_ratio':mean('variance_ratio'),'mean_token_ratio':mean('token_ratio'),
  'per_node':summaries}
 rendered=json.dumps(aggregate,indent=2)
 if args.out: Path(args.out).write_text(rendered+'\n')
 print(rendered)
if __name__=='__main__': main()

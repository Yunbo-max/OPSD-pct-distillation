#!/usr/bin/env python3
"""Summarize PRM-vs-terminal node and branch correlations."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def rank(values):
 order=np.argsort(values,kind='mergesort'); result=np.empty(len(values)); result[order]=np.arange(len(values)); return result
def corr(a,b):
 a=np.asarray(a); b=np.asarray(b)
 return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else float('nan')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('input'); ap.add_argument('--out'); args=ap.parse_args()
 rows=json.loads(Path(args.input).read_text())['nodes']; prm=[]; terminal=[]
 for row in rows:
  prm.append(float(np.mean(row['prm_differences']))); terminal.append(float(np.mean(row['terminal_differences'])))
 pair_prm=np.concatenate([row['prm_differences'] for row in rows]); pair_terminal=np.concatenate([row['terminal_differences'] for row in rows])
 report={'nodes':len(rows),'pairs':len(pair_prm),'node_pearson':corr(prm,terminal),
  'node_spearman':corr(rank(prm),rank(terminal)),'pair_pearson':corr(pair_prm,pair_terminal),
  'pair_spearman':corr(rank(pair_prm),rank(pair_terminal)),
  'node_sign_agreement':float(np.mean(np.sign(prm)==np.sign(terminal))),
  'pair_sign_agreement':float(np.mean(np.sign(pair_prm)==np.sign(pair_terminal))),
  'prm_advantage_range':[min(prm),max(prm)],'terminal_advantage_range':[min(terminal),max(terminal)],
  'terminal_negative_nodes':int(sum(x < -.01 for x in terminal)),'prm_negative_nodes':int(sum(x < -.01 for x in prm))}
 text=json.dumps(report,indent=2)+'\n'; print(text,end='')
 if args.out: Path(args.out).write_text(text)
if __name__=='__main__': main()

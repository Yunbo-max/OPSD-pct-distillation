#!/usr/bin/env python3
"""Fit a problem-split ridge prefix value model and report held-out calibration."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pct.advantage_estimators import fit_ridge_value_head, predict_ridge_value

def fold(pid: str, seed: int) -> int:
 return int.from_bytes(hashlib.sha256(f'{seed}|{pid}'.encode()).digest()[:4],'big')%5
def auroc(score,target):
 pos=score[target>0.5]; neg=score[target<=0.5]
 if not len(pos) or not len(neg): return float('nan')
 return float(((pos[:,None]>neg[None,:]).float()+.5*(pos[:,None]==neg[None,:]).float()).mean())
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('features'); ap.add_argument('--out',required=True)
 ap.add_argument('--ridge_grid',default='.001,.01,.1,1,10'); ap.add_argument('--seed',type=int,default=20260828)
 args=ap.parse_args(); data=torch.load(args.features,map_location='cpu',weights_only=False)
 x=data['features'].float(); y=data['outcomes'].float(); folds=torch.tensor([fold(str(p),args.seed) for p in data['problem_ids']])
 train=folds!=0; valid=~train; mean=x[train].mean(0); std=x[train].std(0).clamp_min(1e-4); xn=(x-mean)/std
 results=[]; best=None
 for ridge in map(float,args.ridge_grid.split(',')):
  w=fit_ridge_value_head(xn[train],y[train],ridge=ridge); pred=predict_ridge_value(xn[valid],w)
  result={'ridge':ridge,'brier':float((pred-y[valid]).square().mean()),'auroc':auroc(pred,y[valid]),
          'mean_prediction':float(pred.mean()),'base_rate':float(y[valid].mean())}; results.append(result)
  if best is None or result['brier']<best[0]: best=(result['brier'],ridge,w)
 bundle={'weights':best[2],'feature_mean':mean,'feature_std':std,'ridge':best[1],'validation':results,
         'source_features':str(Path(args.features).resolve()),'model':data['model'],'split_seed':args.seed}
 out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); torch.save(bundle,out)
 print(json.dumps({'selected_ridge':best[1],'validation':results},indent=2),flush=True)
if __name__=='__main__': main()

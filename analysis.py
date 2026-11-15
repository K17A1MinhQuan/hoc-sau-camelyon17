"""Camelyon17 final analysis: Q1/Q2/Q3; does not train or modify metrics."""
import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ['erm', 'erm_aug', 'group_dro', 'group_dro_aug']
LABELS = {'erm':'ERM', 'erm_aug':'ERM + Aug', 'group_dro':'Group DRO', 'group_dro_aug':'Group DRO + Aug'}
HOSPITALS = [0, 1, 3, 4]
RISK = ['avg_risk_val4', 'worst_risk_val4']
HACC = [f'acc_val_h{x}' for x in HOSPITALS]
COLORS = {'Train':'#16a34a', 'ID validation':'#2563eb', 'OOD validation':'#ea580c'}


def numeric(df, columns):
    for c in columns:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def validate(df):
    for c in ['model', 'seed']:
        if c not in df:
            raise ValueError(f'Missing required column: {c}')
    if df.duplicated(['model', 'seed']).any():
        raise ValueError('Duplicate model/seed rows in metrics.csv; resolve before analysis.')
    return numeric(df, ['id_accuracy','ood_accuracy','id_auroc','ood_auroc','id_loss','ood_loss',
                        'test_accuracy','test_auroc',*RISK,*HACC])


def save(fig, path):
    fig.savefig(path, dpi=190, bbox_inches='tight')
    plt.close(fig)
    print('Saved:', path.name)


def bar_chart(df, cols, labels, title, ylabel, dest, ylim=None):
    means = df.groupby('model')[cols].mean().reindex(MODELS)
    counts = df.groupby('model')[cols].count().reindex(MODELS).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(MODELS))
    width = 0.8 / len(cols)
    palette = ['#2563eb', '#ea580c', '#16a34a', '#a855f7', '#64748b']
    values = means.to_numpy(dtype=float)
    for j, col in enumerate(cols):
        pos = x - .4 + width*(j+.5)
        vals = values[:,j]
        bars = ax.bar(pos, np.nan_to_num(vals, nan=0), width*.92,
                      color=palette[j], label=labels[j])
        for i, b in enumerate(bars):
            n = counts.iloc[i,j]
            if n == 0:
                b.set_alpha(0)
                ax.text(b.get_x()+b.get_width()/2, 0.015 if ylim else 0,
                        'N/A', ha='center', va='bottom', fontsize=8, rotation=90)
            else:
                ax.annotate(f'{vals[i]:.3f}\n(n={n})',
                            (b.get_x()+b.get_width()/2,b.get_height()),
                            xytext=(0,3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x, [LABELS[m] for m in MODELS])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        mx = np.nanmax(values) if np.isfinite(values).any() else 1
        ax.set_ylim(0, max(.1, mx*1.25))
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.12), ncol=len(cols), frameon=False)
    ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    save(fig, dest)


def q2_chart(df, root):
    pairs = [('erm','erm_aug'), ('group_dro','group_dro_aug')]
    fig, ax = plt.subplots(figsize=(8.6,5))
    for i,(base,aug) in enumerate(pairs):
        a = df[df.model==base][['seed','ood_accuracy']].rename(columns={'ood_accuracy':'a'})
        b = df[df.model==aug][['seed','ood_accuracy']].rename(columns={'ood_accuracy':'b'})
        merged = a.merge(b,on='seed').dropna()
        if merged.empty:
            ax.text(i,0,'N/A',ha='center',va='bottom')
            continue
        delta = (merged.b-merged.a)*100
        mean = delta.mean()
        ax.bar(i,mean,color='#16a34a' if mean>=0 else '#ea580c',width=.55)
        ax.text(i,mean, f'{mean:+.2f} pp\n(n={len(delta)})',ha='center',
                va='bottom' if mean>=0 else 'top',fontsize=10)
        print(f'Q2 {base} -> {aug}: mean OOD delta={mean:+.3f} pp, paired seeds={len(delta)}')
    ax.axhline(0,color='#64748b',lw=1)
    ax.set_xticks([0,1], ['ERM + Aug vs ERM','Group DRO + Aug vs Group DRO'])
    ax.set_ylabel('OOD accuracy change (percentage points)')
    ax.set_title('Q2: Paired augmentation effect on OOD validation')
    ax.grid(axis='y',alpha=.15)
    fig.tight_layout()
    save(fig,root/'q2_augmentation_ood_delta.png')


def loss_charts(root):
    grouped = {m: {'Train':[], 'ID validation':[], 'OOD validation':[]} for m in MODELS}
    pattern = re.compile(r'^(erm|erm_aug|group_dro|group_dro_aug)_seed(\d+)_history\.json$')
    for path in sorted(root.glob('*_history.json')):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        try:
            h = json.loads(path.read_text(encoding='utf-8'))
            series = {'Train':h.get('train_loss',[]),
                      'ID validation':[v.get('loss',np.nan) for v in h.get('id_val',[])],
                      'OOD validation':[v.get('loss',np.nan) for v in h.get('ood_val',[])]}
            for key, values in series.items():
                arr = np.asarray(values,dtype=float)
                if arr.size:
                    grouped[match.group(1)][key].append(arr)
        except (ValueError,TypeError,OSError,KeyError,AttributeError) as e:
            warnings.warn(f'Invalid history {path.name}: {e}')
    fig,axes = plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
    all_values=[]
    for ax,model in zip(axes.flat,MODELS):
        ax.set_title(LABELS[model])
        for key,arrays in grouped[model].items():
            if not arrays:
                continue
            length = max(map(len,arrays))
            matrix = np.full((len(arrays),length),np.nan)
            for i,a in enumerate(arrays):
                matrix[i,:len(a)] = a
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                mean = np.nanmean(matrix,axis=0)
                std = np.nanstd(matrix,axis=0,ddof=1) if len(arrays)>1 else np.zeros(length)
            valid = np.isfinite(mean)
            epochs = np.arange(1,length+1)
            ax.plot(epochs[valid],mean[valid],color=COLORS[key],lw=2,label=f'{key} (n={len(arrays)})')
            if len(arrays)>1:
                lower=np.maximum(0,mean-std)
                ax.fill_between(epochs,lower,mean+std,where=valid,color=COLORS[key],alpha=.10)
            all_values.extend(mean[valid].tolist())
        if not any(grouped[model].values()):
            ax.text(.5,.5,'No history available',transform=ax.transAxes,ha='center')
        ax.grid(alpha=.18)
        if ax.lines:
            ax.legend(fontsize=8,loc='upper right')
    if all_values:
        high=np.nanpercentile(all_values,99.5)
        axes[0,0].set_ylim(0,max(.1,high*1.2))
    for ax in axes[-1,:]: ax.set_xlabel('Epoch')
    for ax in axes[:,0]: ax.set_ylabel('Cross-entropy loss')
    fig.suptitle('Q1: Training and validation loss by model (mean across seeds; shaded ±1 SD)')
    fig.tight_layout()
    save(fig,root/'q1_loss_curves.png')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results-dir',default='results/corrected_v1')
    args=p.parse_args()
    root=Path(args.results_dir)
    df=validate(pd.read_csv(root/'metrics.csv'))
    summary=[]
    for m in MODELS:
        sub=df[df.model==m]
        if sub.empty:
            warnings.warn(f'No metrics for {m}')
            continue
        gaps=(sub.id_accuracy-sub.ood_accuracy)*100
        row={'model':m,'n_seeds':len(sub),'id_accuracy_mean':sub.id_accuracy.mean(),
             'ood_accuracy_mean':sub.ood_accuracy.mean(),
             'id_ood_gap_pp_mean':gaps.mean(),
             'avg_risk_val4_mean':sub.avg_risk_val4.mean(),
             'worst_risk_val4_mean':sub.worst_risk_val4.mean(),
             'q3_valid_seeds':int(sub[RISK+HACC].notna().all(axis=1).sum())}
        summary.append(row)
        print(f"{m}: Q1 gap={row['id_ood_gap_pp_mean']:.2f} pp; Q3 complete={row['q3_valid_seeds']}/{len(sub)} seeds")
    pd.DataFrame(summary).to_csv(root/'analysis_summary.csv',index=False)
    missing=df[['model','seed',*RISK,*HACC]].copy()
    missing['q3_complete']=missing[RISK+HACC].notna().all(axis=1)
    missing[['model','seed','q3_complete']].to_csv(root/'q3_data_completeness.csv',index=False)
    if not missing.q3_complete.all():
        print('WARNING: Q3 is incomplete. Missing results are labeled N/A, not zero.')
        print(missing.loc[~missing.q3_complete,['model','seed']].to_string(index=False))
    bar_chart(df,['id_accuracy','ood_accuracy'],['ID validation','OOD validation'],
              'Q1: ID vs OOD validation accuracy','Accuracy',root/'q1_id_ood_accuracy.png',ylim=(0,1.12))
    q2_chart(df,root)
    if missing.q3_complete.any():
        bar_chart(df,RISK,['Average risk','Worst-domain risk'],
                  'Q3: Validation risk (H0, H1, H3, H4; excludes test H2)',
                  'Cross-entropy risk',root/'q3_validation_risk.png')
        bar_chart(df,HACC,['H0 (ID)','H1 (OOD)','H3 (ID)','H4 (ID)'],
                  'Q3: Validation accuracy by hospital (excludes test H2)',
                  'Accuracy',root/'q3_validation_hospital_accuracy.png',ylim=(0,1.15))
    else:
        print('WARNING: No Q3 validation results; skipping Q3 figures.')
    all5=[f'acc_h{i}' for i in range(5)]
    if all(c in df.columns for c in all5):
        numeric(df,all5)
        bar_chart(df,all5,[f'H{i}' + (' (TEST)' if i==2 else '') for i in range(5)],
                  'Post-hoc only: all five hospitals (includes held-out test H2)',
                  'Accuracy',root/'q3_all5_posthoc_accuracy.png',ylim=(0,1.15))
    loss_charts(root)
    print('Completed. Source metrics.csv was not modified.')

if __name__=='__main__':
    main()

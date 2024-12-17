import io
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
import scipy
from tqdm.autonotebook import tqdm
import glob

import captum


"""
Code from: https://arxiv.org/abs/2312.02186
"""

class ModelWrapper(torch.nn.Module):
    """Wraps a model with multiple outputs into a single output model specified by a specific
    target string that corresponds to an element of model.targets. The model then has a .targets 
    list that only has one element and the output of the model will have only one entry.
    A softmax or sigmoid can be applied if specified or a `adjustment` which can be a lambda function
    that will be applied to the output of the model.

    `target` can also be a list of tuples (coefficient, target) that will be composed together to form an
    additive classifier with a single output. The output of the model for each target will be composed 
    together after they are multiplied by their coefficient.

    The string representation is dynamically generated to help with logging experiments with the form
    ModelName (target name). This can be overwritten. `rename` will rename the target name specified 
    and `rename_model` will rename the model. Setting `rename_model` can be set to the empty string to 
    just print the target name.
    
    """
    def __init__(
        self, 
        clf, 
        target, 
        softmax: bool = False, 
        sigmoid: bool = False, 
        adjustment=None,
        rename: str=None,
        rename_model: str=None,
    ):
        super().__init__()
        self.clf = clf
        self.device = next(self.clf.parameters()).device
        self.target = target
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.adjustment = adjustment
        self.rename = rename
        self.rename_model = rename_model
        
        if self.rename is None:
            if type(self.target) == list:
                self.targets = [''.join([f'{"" if coef == 1 else "+"+str(coef) if coef > 0 else coef}{target}' for target, coef in self.target])]
            else:
                self.targets = [self.target]
        else:
            self.targets = [self.rename]
    
    def __repr__(self):
        if self.rename_model is None:
            return f'{self.clf} ({self.targets[0]})'
        elif self.rename_model == '':
            return self.targets[0]
        else:
            return f'{self.rename_model} ({self.targets[0]})'
            
    def forward(self, x):
        output = self.clf(x)
        if not self.adjustment is None:
            output = self.adjustment(output)
        if self.softmax:
            output = torch.softmax(output,1)

        if type(self.target) == list:
            result = torch.zeros(1).to(self.device)
            
            for target,coef in self.target:
                
                thisoutput = output[:,[self.clf.targets.index(target)]]
                
                if self.sigmoid:
                    thisoutput = torch.sigmoid(thisoutput)
                
                result = result + thisoutput * coef
            
        else:
            result = output[:,[self.clf.targets.index(self.target)]]
            if self.sigmoid:
                result = torch.sigmoid(result)
        return result


def compute_rchange(model1_preds, ext_preds):
    base_change = (model1_preds[-1] - model1_preds[0]) 
    ext_change = (ext_preds[-1] - ext_preds[0])
    if base_change == ext_change:
        return 1.0
    else:
        return ext_change/(base_change+1E-20)

def compute_alignment(
    x: np.ndarray,
    model1,
    models: list,
    ae,
    return_output=False,
    seperate_models=False,
    lambda_sweep_steps=2, # 2 is all that is needed to compute rchange but not good for plots
):
    """Computes the alignment between classifier outputs for a counterfactual image.
    
    
    seperate_models: If false the average between all models is computed. If true then
        each model is computed against `model1` and multiple results are returned. This
        is for efficiency because the CF generation process is costly so we can compare 
        multiple classifiers while only computing one CF.
        
    """
    
    device = next(model1.parameters()).device

    x = torch.from_numpy(x).unsqueeze(0).to(device)

    attr = captum.attr.LatentShift(model1, ae)
    output = attr.attribute(x, target=0,
                            #fix_range=[-2400,0],
                            return_dicts = True,
                            search_pred_diff = 0.6,
                            verbose=False,
                            apply_sigmoid=False,
                            search_max_steps=500000,
                            lambda_sweep_steps=lambda_sweep_steps,
                           )[0]
    
    

    return compute_alignment_from_output(
        output = output,
        model1 = model1,
        models = models,
        ae = ae,
        return_output=return_output,
        seperate_models=seperate_models,
    )


def compute_alignment_from_output(
    output: dict,
    model1,
    models: list,
    ae,
    return_output=False,
    seperate_models=False,
):
    """Computes the alignment between classifier outputs for a counterfactual image.
    
    
    seperate_models: If false the average between all models is computed. If true then
        each model is computed against `model1` and multiple results are returned. This
        is for efficiency because the CF generation process is costly so we can compare 
        multiple classifiers while only computing one CF.
        
    """
    
    device = next(model1.parameters()).device

    base_clf_name = f'*{model1}'
    base_clf_preds = []
    for img in output['generated_images']:
        xp = torch.from_numpy(img[None,...]).to(device)
        out = model1(xp)[0, 0].detach().cpu().numpy()
        base_clf_preds.append(float(out))
    
    results = []

    #clunky but supports the two use cases well
    if seperate_models:
        models = [[m] for m in models]
    else:
        models = [models]

    for target_models in models:
        result = {}
        #result['lambdas'] = list(output['lambdas'])

        if seperate_models:
            result['model1_target'] = model1.targets[0]
            result['models_target'] = target_models[0].targets[0]
            

        preds = {}
        preds[base_clf_name] = list(base_clf_preds)
        
        for m in target_models:
            this_clf_name = f'{m}'
            preds[this_clf_name] = []
            for img in output['generated_images']:
                xp = torch.from_numpy(img[None,...]).to(device)
                out = m(xp)[0, 0].detach().cpu()
                preds[this_clf_name].append(round(float(out.numpy()),4))
            
        result['preds'] = preds
        
        focus = base_clf_name
        # pearsonrs = [scipy.stats.pearsonr(preds[focus], preds[k]).statistic for k in preds if focus != k]
        # result['pearsonrs'] = np.mean(pearsonrs).round(4)
            
        rchange = [compute_rchange(preds[focus], preds[k]) for k in preds if focus != k]
        result['rchange'] = np.mean(rchange).round(4)
    
        diffs = [np.ptp(preds[k],0) for k in preds if focus != k]
        result['diffs'] = np.mean(diffs).round(4)
        
        # kendalltaus = [scipy.stats.kendalltau(preds[focus], preds[k]).correlation for k in preds if focus != k]
        # result['kendalltaus'] = np.mean(kendalltaus).round(4)
        
        # spearmanrs = [scipy.stats.spearmanr(preds[focus], preds[k]).correlation for k in preds if focus != k]
        # result['spearmanrs'] = np.mean(spearmanrs).round(4)
    
        if not return_output:
            results.append(result)
        else:
            results.append([result, output])

    if seperate_models:
        return results
    else:
        assert len(results) == 1
        return results[0]

def compute_basechange(preds):
    if type(preds) == str:
        preds = eval(preds)
    model1_preds = list(preds.values())[0]
    base_change = (model1_preds[-1] - model1_preds[0])
    return base_change


def make_pretty(styler):
    """Style for pandas tables for heatmaps and more."""
    styler.format(lambda v: f'{v:.2f}')
    styler.background_gradient(axis=None, vmin=-1, vmax=1, cmap="RdBu")
    styler.set_table_styles(
    [dict(selector="th.col_heading",props=[
        ("transform", "rotate(180deg)"),
        ("text-align", "left"),
        ("writing-mode", "vertical-lr"),
    ])])
    return styler


def compute_matrix(
    df,
    cols: list = None,
    rows: list = None,
    base_change_limit: float = 0.3,
    restrict_rows: bool = True,
    style: bool = False,
    target: str = 'rchange',
    with_stderr: bool = False,
    precision: int = 3,
):
    """Compute an alignment matrix from an alignment dataframe.
    `cols` is the list of columns to plot. The matrix will be ordered using this 
    so the diagonal will be the intersection.
    """
    df = df.copy()

    if base_change_limit > 0:
        df['base_change'] = df.preds.apply(compute_basechange)
        df = df[(df['base_change'] > base_change_limit)]
        if len(df) == 0:
            print('Not enough changes')
    
    if restrict_rows:
        if cols is None:
            cols = df.models_target.unique()
        
        grouped = df[df.model1_target.isin(cols)].groupby(['model1_target','models_target'])[[target]]
        
        g_agg = grouped.mean().unstack(1).droplevel(0,1)#.round(2)
        g_agg = g_agg.loc[cols][cols]
        
        g_agg_stderr = grouped.std().unstack(1).droplevel(0,1)
        g_agg_stderr = (g_agg_stderr/grouped.count().unstack(1).droplevel(0,1).map(np.sqrt))#.round(2)
        g_agg_stderr = g_agg_stderr.loc[cols][cols]
        
    else:
        grouped = df.groupby(['model1_target','models_target'])[[target]]
        g_agg = grouped.mean().unstack(1).droplevel(0,1)#.round(4)
        g_agg_stderr = grouped.std().unstack(1).droplevel(0,1)
        g_agg_stderr = (g_agg_stderr/grouped.count().unstack(1).droplevel(0,1).map(np.sqrt))#.round(2)

    if with_stderr:
        stdevs = g_agg_stderr.values.flatten().tolist()
        stdevs.reverse()
        g_agg = g_agg.map(lambda x: f'{x:.{precision}f}±{stdevs.pop():.{precision}f}')
    
    if rows is not None:
        g_agg = g_agg.loc[rows]
    if cols is not None:
        g_agg = g_agg[cols]

    if style:
        g = g_agg.style.pipe(make_pretty)
        g.index.name = None
        g.columns.name = None
        return g
    else:
        return g_agg

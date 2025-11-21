"""
"""
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import os
import scanpy as sc

from tqdm import tqdm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ValueWarning


def run_lmm(mat, genes, obs, obs_fixed, obs_random, output=None, offset=1e-2, min_max_expr_th=0.1):
    """
     mat - cell by gene - cp10k norm
     genes - gene names in mat
     obs - cell names and other metadata in mat
     
    """
    
    c0, c1 = np.unique(obs[obs_fixed])
    obs = obs[[obs_fixed, obs_random]]
    zmat = stats.zscore(np.log2(mat+1), axis=0)
    print(zmat.shape, obs.shape)
    
    # remove genes that are NaN in Zmat (no variation at all; or all zero)
    cond_nan = np.any(np.isnan(zmat), axis=0)
    mat = mat[:,~cond_nan]
    zmat = zmat[:,~cond_nan]
    genes = genes[~cond_nan]
    genes_idx = np.arange(len(genes)) # local index
    print(zmat.shape, obs.shape)

    # remove genes that do not satisfy min cond_expr
    df = pd.DataFrame(mat, columns=np.char.add('g', genes_idx.astype(str)), index=obs.index)
    df = df.join(obs) # .dropna()
    df_mean_sample = df.groupby([obs_random]).mean(numeric_only=True)
    df_mean_sample.columns = genes_idx
    cond_expr = (df_mean_sample.max() > min_max_expr_th).values
    
    mat = mat[:,cond_expr]
    zmat = zmat[:,cond_expr]
    genes = genes[cond_expr]
    genes_idx = np.arange(len(genes)) # local index
    print(zmat.shape, obs.shape)
    
    # cp10k scale, ctrds and log2fc (fast)
    df = pd.DataFrame(mat, columns=np.char.add('g', genes_idx.astype(str)), index=obs.index)
    df = df.join(obs) # .dropna()
    
    df_mean_sample = df.groupby([obs_random]).mean(numeric_only=True)
    df_mean_sample.columns = genes_idx
    cond_expr = (df_mean_sample.max() > min_max_expr_th).values
    assert np.all(cond_expr)
    

    df_mean = df.groupby([obs_fixed]).mean(numeric_only=True)
    log2fc  = ( np.log2(df_mean.loc[c1]+offset)
               -np.log2(df_mean.loc[c0]+offset)).values
    cond_fc = (np.abs(log2fc) > np.log2(2))
    print(cond_fc.sum(), genes[cond_fc])

    # zscore(log2(cp10k)) scale [zscore is needed for the mixedlm model to converge], test (slow)
    zdf = pd.DataFrame(zmat, columns=np.char.add('g', genes_idx.astype(str)), index=obs.index)
    zdf = zdf.join(obs) # .dropna()

    # formal test (slow)
    pvals = []
    converges = []
    for i in tqdm(genes_idx):
        model = smf.mixedlm(f"g{i} ~ {obs_fixed}", zdf, groups=obs_random)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            # warnings.simplefilter("ignore", convergencewarning)
            # warnings.simplefilter("ignore", runtimewarning)
            try:
                result = model.fit()
                pval = result.pvalues[f'{obs_fixed}[T.{c1}]']
                converged = result.converged
            except:
                pval = 1
                converged = False


        pvals.append(pval)
        converges.append(converged)

    converges = np.array(converges)
    pvals = np.nan_to_num(np.array(pvals), 1)
    rej, qvals, _, _ =  multipletests(pvals, alpha=0.05, method='fdr_bh')
    cond_all = np.logical_and(cond_fc, rej)

    print(cond_all.sum(), genes[cond_all])
    # save results: exp_cond, subclass, genes, log2fc, qvals

    df_res = pd.DataFrame(index=genes_idx)
    df_res['gene'] = genes
    df_res['log2fc'] = log2fc
    df_res['pval'] = pvals
    df_res['qval'] = qvals
    df_res['converged'] = converges
    df_res = df_res.join(df_mean_sample.T)
    if output is not None:
        print(output)
        df_res.to_csv(output)
        
    return df_res



def run_lmm_two_fixed(mat, genes, obs, obs_fixed1, obs_fixed2, obs_random, output=None, offset=1e-2, min_max_expr_th=0.1):
    """
     mat - cell by gene - cp10k norm
     genes - gene names in mat
     obs - cell names and other metadata in mat
     
    """
    
    obs = obs[[obs_fixed1, obs_fixed2, obs_random]]
    ymat = np.log2(mat+1)
    sigmas = np.std(ymat, axis=0) # used to recover log2FC
    zmat = stats.zscore(ymat, axis=0)
    print(zmat.shape, obs.shape)
    
    # remove genes that are NaN in Zmat (no variation at all; or all zero)
    cond_nan = np.any(np.isnan(zmat), axis=0)
    mat = mat[:,~cond_nan]
    zmat = zmat[:,~cond_nan]
    genes = genes[~cond_nan]
    sigmas = sigmas[~cond_nan]
    genes_idx = np.arange(len(genes)) # local index
    print(zmat.shape, obs.shape)

    # remove genes that do not satisfy min cond_expr
    df = pd.DataFrame(mat, columns=np.char.add('g', genes_idx.astype(str)), index=obs.index)
    df = df.join(obs) # .dropna()
    df_mean_sample = df.groupby([obs_random]).mean(numeric_only=True)
    df_mean_sample.columns = genes_idx
    cond_expr = (df_mean_sample.max() > min_max_expr_th).values
    
    mat = mat[:,cond_expr]
    zmat = zmat[:,cond_expr]
    genes = genes[cond_expr]
    sigmas = sigmas[cond_expr]
    genes_idx = np.arange(len(genes)) # local index
    print(zmat.shape, obs.shape)
    

    # zscore(log2(cp10k)) scale [zscore is needed for the mixedlm model to converge], test (slow)
    zdf = pd.DataFrame(zmat, columns=np.char.add('g', genes_idx.astype(str)), index=obs.index)
    zdf = zdf.join(obs) # .dropna()

    # formal test (slow)
    pvals = []
    params = []
    converges = []
    for i in tqdm(genes_idx):
        model = smf.mixedlm(f"g{i} ~ {obs_fixed1} * {obs_fixed2}", zdf, groups=obs_random)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            # warnings.simplefilter("ignore", convergencewarning)
            # warnings.simplefilter("ignore", runtimewarning)
            result = model.fit()
        
        pval = result.pvalues
        param = result.params
        converged = result.converged

        pvals.append(pval)
        params.append(param)
        converges.append(converged)
    
    labels = params[0].index.values
    params = pd.concat(params, axis=1).T.values #.T
    pvals  = pd.concat(pvals, axis=1).T.values  #.T
    converges = np.array(converges)
    
    pvals = np.nan_to_num(pvals, 1)
    rej, qvals, _, _ =  multipletests(pvals.reshape(-1,), alpha=0.05, method='fdr_bh')
    qvals = qvals.reshape(pvals.shape)
    
    res = sc.AnnData(X=pvals, obs=pd.DataFrame(index=genes), var=pd.DataFrame(index=labels))
    res.obs['sigma'] = sigmas
    res.obs['converge'] = converges
    res.layers['pval'] = pvals
    res.layers['param'] = params
    res.layers['effsize'] = sigmas.reshape(-1,1)*params
    res.layers['qval'] = qvals
    
    if output is not None: 
        res.write(output)
    
    return res

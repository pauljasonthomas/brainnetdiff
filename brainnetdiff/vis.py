#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2021--, CoNeCT Lab, University of Illinois at Chicago .

Distributed under the terms of the Modified BSD License.

The full license is in the file COPYING.txt, distributed with this software.

NOTES:

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn.plotting import plot_connectome
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


def ctm_plot2group(bnd, nbs_idx, xyz, title='', g1_title='HC',
                   g2_title='PT', path=None, catplot=['roi','region'],
                   make_colorbars=True, barplot_type=None,
                   node_color='black',
                   node_cmap='magma',
                   node_size=50,
                   node_kwargs={'alpha': 0.75, 'rasterized':True},
                   edge_cmap='magma',
                   edge_kwargs={'linewidth': 3}):
    """Method for creating plots for visualizing NBS subnetworks using SDD connectome analyses as in Thomas et al., 2021. Note that because this creates one set of brain plots for each of two groups, it only works when following the nbs_ttest() method. ***Also requires modified nilearn plot_connectome() method to allow for passage of normalized edge colormaps.

    Parameters
    ----------
    bnd : brainnetdiff object
        brainnetdiff object
    nbs_idx : int
        specifies the subnetwork to use for diffusion analysis as indexed in the column 'nbs_idx' of the pandas dataframe obtained using the 'get_nbs_df()' method
    xyz : numpy array
        array of shape (N, 3) containing the xyz coordinates corresponding to rois supplied in the parameter roi_names of the brainnetdiff object.
    title : string
        suptitle for plot
    g1_title : string
        subtitle for brain plots for group 1 from nbs_ttest(). Default is 'HC'.
    g2_title : string
        subtitle for brain plots for group 2 from nbs_ttest(). Default is 'PT'.
    path : string
        filepath+name for saving plots. If None (default) plots will not be saved.
    catplot : {list, None}
        list of strings that specifies which types of seaborn catplot to make to display strength of nodes in the SDD connectomes. Use 'region' to group nodes by region, 'roi' to display data for individual rois, and None to skip generation of these barplots. Default (['roi', 'region']) generates plots for both.
    make_colorbars : boolean
        True (default) will also generate colorbars/legends for plot
    node_color : {matplotlib color, 'strength'}
        specifies how to color nodes. Use a string that represents a matplotlib color (default: 'black') for uniform node style or 'strength' to color nodes by SDD strength in the subnetwork.
    node_cmap : {matplotlib colormap, None}
        specifies colormap for coloring nodes in connectome and/or barplots (default: 'magma').
    node_size : int
        specifies size of nodes - passed to nilearn plot_connectome() method (default: 50).
    node_kwargs : dict
        passed to nilearn plot_connectome() method
    edge_cmap : {matplotlib colormap, None}
        specifies colormap for coloring edges in connectome plots (default: 'magma'). Passed to modified nilearn plot_connectome() method.
    edge_kwargs : dict
        passed to nilearn plot_connectome() method

    Returns
    -------
    self : returns an instance of self
    """
    if bnd.nbs_type == 'cor':
        raise ValueError('Method only works with and following nbs_ttest() method, sorry :(')
    g1_ind = bnd.g1_idx
    g2_ind = bnd.g2_idx
    plot_idx = bnd.nbs_res_data[nbs_idx]
    adj = plot_idx[1]
    n_ind = plot_idx[3]
    n_roi = xyz[n_ind,:]
    n_adj = adj[np.ix_(n_ind, n_ind)]
    n_adj[n_adj > 0] = 1
    SDD = np.copy(bnd.SDD)
    A = SDD[np.ix_(n_ind, n_ind, np.arange(SDD.shape[2]))]
    A[n_adj == 0] = 0
    A1 = A[:,:,g1_ind].mean(axis = 2)
    A2 = A[:,:,g2_ind].mean(axis = 2)
    if node_color == 'strength':
        nvals1 = A1.sum(axis = 1)
        nvals2 = A2.sum(axis = 1)
        vmin=np.min((nvals1.min(), nvals2.min()))
        vmax=np.min((nvals1.max(), nvals2.max()))
        node_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        node_kwargs['norm'] = node_norm
        node_kwargs['cmap'] = node_cmap
    elif node_color == 'cortex':
        print('in development... sorry')
        return
    else:
        nvals1 = node_color
        nvals2 = node_color
    if edge_cmap is not None:
        emin=np.min([A1.min(), A2.min()])
        emax=np.max([A1.max(), A2.max()])
        edge_norm = mpl.colors.Normalize(vmin=emin, vmax=emax)
    else:
        edge_norm = None
    f, ax = plt.subplots(nrows = 2, figsize=(15,10))
    plot_connectome(A1, node_coords=n_roi, node_size=node_size,
                     node_color=nvals1, node_kwargs=node_kwargs,
                     colorbar=False, axes = ax[0],
                     edge_kwargs=edge_kwargs, edge_cmap=edge_cmap,
                     edge_norm=edge_norm, title = None)
    plot_connectome(A2, node_coords=n_roi, node_size=node_size,
                     node_color=nvals2, node_kwargs=node_kwargs,
                     colorbar=False, axes = ax[1],
                     edge_kwargs=edge_kwargs, edge_cmap=edge_cmap,
                     edge_norm=edge_norm, title = None)
    fd = {'fontsize':16, 'fontweight':'bold'}
    ax[0].set_title(g1_title, loc='left', fontdict=fd)
    ax[1].set_title(g2_title, loc='left', fontdict=fd)
    f.suptitle(title, fontsize = 24)
    if path is not None:
        plt.savefig(path+'nbs_sn_ctm.png',
                    dpi = 300,
                    bbox_inches = 'tight',
                    facecolor='white')
    plt.show()
    if make_colorbars:
        if node_color == 'strength' or node_color == 'cortex':
            fig, ax = plt.subplots(1, 1)
            SM = mpl.cm.ScalarMappable(norm=node_norm, cmap=node_cmap)
            cbar = ax.figure.colorbar(SM, ax=ax, pad=.05, orientation = 'horizontal')
            ax.axis('off')
            if path is not None:
                plt.savefig(path+'nbs_sn_ctm_node_colorbar.png',
                            dpi = 300,
                            bbox_inches = 'tight',
                            facecolor='white')
            plt.show()
        elif edge_cmap is not None:
            fig, ax = plt.subplots(1, 1)
            SM = mpl.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
            cbar = ax.figure.colorbar(SM, ax=ax, pad=.05, orientation = 'horizontal')
            ax.axis('off')
            if path is not None:
                plt.savefig(path+'nbs_sn_ctm_edge_colorbar.png',
                            dpi = 300,
                            bbox_inches = 'tight',
                            facecolor='white')
            plt.show()

    if catplot is not None:
        nvals = np.concatenate((A1.sum(axis = 1), A2.sum(axis = 1)))
        if 'roi' in catplot:
            node_names = plot_idx[4]*2
            group = ['HC']*A1.shape[0]+['PT']*A2.shape[0]
            degdf = pd.DataFrame(list(zip(node_names, nvals, group)),
                                 columns = ['node', 'mean_strength', 'group'])
            degdf = degdf.sort_values('mean_strength',ascending=False).reset_index()
            f, ax = plt.subplots(figsize = (2,6))
            sns.barplot(data = degdf, x = 'mean_strength', y = 'node',
                                hue = 'group', ax=ax, palette="dark")
            sns.despine()
            ax.legend(loc='lower right')
            plt.setp(ax.get_yticklabels(), fontsize=6.5)
            plt.ylabel(None)
            plt.xlabel('mean '+r'$\mathrm{strength}_{SDD}$')
            if path is not None:
                plt.savefig(path+'roi'+'_grouped_barplot.png', dpi = 300,
                            bbox_inches = 'tight',
                            facecolor='white')
            plt.show()
        if 'region' in catplot:
            node_names = plot_idx[5]*2
            group = ['HC']*A1.shape[0]+['PT']*A2.shape[0]
            degdf = pd.DataFrame(list(zip(node_names, nvals, group)),
                                 columns = ['node', 'mean_strength', 'group'])
            degdf = degdf.sort_values('mean_strength',ascending=False).reset_index()
            f, ax = plt.subplots(figsize = (2,6))
            sns.barplot(data = degdf, x = 'mean_strength', y = 'node',
                                hue = 'group', ax=ax, palette="dark")
            sns.despine()
            ax.legend(loc='lower right')
            plt.setp(ax.get_yticklabels(), fontsize=6.5)
            plt.ylabel(None)
            plt.xlabel('mean '+r'$\mathrm{strength}_{SDD}$')
            if path is not None:
                plt.savefig(path+'region'+'_grouped_barplot.png', dpi = 300,
                            bbox_inches = 'tight',
                            facecolor='white')
            plt.show()
    return

def hk_cor_plot(bnd, cordf, stat_idx, xyz, trait, trait_name='',
                subset_ind=None, path=None, regplot_titles=None,
                plt_ctm=True, plt_ctm_sq=True, rm_outlier=True, figsize=(10,3)):
    """Method for creating plots for visualizing NBS subnetworks using SDD connectome analyses as in Thomas et al., 2021. Note that because this creates one set of brain plots for each of two groups, it only works when following the nbs_ttest() method. ***Also requires modified nilearn plot_connectome() method to allow for passage of normalized edge colormaps.

    Parameters
    ----------
    bnd : brainnetdiff object
        brainnetdiff object
    cordf : pandas dataframe
        dataframe obtained by get_correlation_df()
    stat_idx : list
        list of integer indices that specifies the rows of the dataframe obtained via get_correlation_df() for which to make plots.
    xyz : numpy array
        array of shape (N, 3) containing the xyz coordinates corresponding to rois supplied in the parameter roi_names of the brainnetdiff object.
    trait : array-like
        array or list of values to correlate with heat kernel values. Must be of length S or len(subset_ind), if given.
    subset_ind : array-like or None
        specifies the indicies of subjects to be used for correlations. Must be of same length as trait. Default (None) assumes all subjects are used.
    path : string
        filepath+name for saving plots. If None (default) plots will not be saved.
    regplot_title : list
        list of strings of titles for correlation plots corresponding to stat_idx. Default (None) results in no titles.
    plt_ctm : Boolean
        specifies whether or not to generate connectome plots where edge colors correspond to the color of correlation plots of corresponding heat kernel values.
    plt_ctm_sq : Boolean
        specifies whether figsize of connectome plots is square (default: True).
    rm_outlier : boolean
        specifies whether to remove largest outlier from data for plots. Note that spearman correlations (which are largely insensitive to outliers) are recalculated and displayed with plots if value is True (default).
    figsize : tuple
        specifies the figzise of the plots. Default is (10,3).

    Returns
    -------
    self : returns an instance of self
    """
    if isinstance(stat_idx, int):
        idx = [stat_idx]
    else:
        idx = [i for i in stat_idx]
    if regplot_titles is None:
        rpt = ['' for i in idx]
    else:
        rpt = regplot_titles
    ncol = len(idx)
    ydist = len(idx)*0.02
    if plt_ctm:
        sn_xyz, sn_names = bnd.avg_reg_xyz(xyz)
        N = len(sn_names)
        A = np.zeros((N,N))
    cols = [mpl.cm.tab10.colors[i] for i in range(len(idx))]
    cm = mpl.colors.ListedColormap(cols)
    bounds = [i+0.5 for i in range(len(idx) + 1)]
    norm = mpl.colors.BoundaryNorm(bounds, cm.N)
    fig, axes = plt.subplots(1, ncol, sharey = True, figsize = figsize)
    for i,ax in enumerate(axes.flat):
        dat = cordf.loc[cordf['stat_idx'] == idx[i]].copy(deep = True)
        i1, i2 = dat.roi1_ind.values[0], dat.roi2_ind.values[0]
        stit = 'rho = {} pval = {} qval = {}'
        stit = stit.format(str(round(dat.stat.values[0], 3)),
                           str(round(dat.pval.values[0], 3)),
                           str(round(dat.qval2.values[0], 3)))
        x = bnd.rHt[i1, i2,:]
        if subset_ind is not None:
            x = x[subset_ind]
        y = trait.copy()
        if rm_outlier:
            xind = np.argmax(x)
            x = np.delete(x, xind)
            y = np.delete(y, xind)
        sns.regplot(x = x, y = y, color = cm(norm(i+1)), ax = ax)
        ax.set_title(rpt[i]+'\n '+stit, fontsize = 8)
        if plt_ctm:
            A[i1,i2], A[i2,i1] = i+1, i+1
    fig.text(0.5, 0.01, 'heat transfer', ha='center')
    fig.text(ydist, 0.5, trait_name, va='center', rotation='vertical')
    if path is not None:
        plt.savefig(path+trait_name.replace(' ','')+'corplot.png',
                    dpi = 300,
                    bbox_inches = 'tight',
                    facecolor='white')
    plt.show()
    if plt_ctm and not plt_ctm_sq:
        fig1, ax = plt.subplots(1, 1, figsize = (6,3))
        plot_connectome(A, node_coords=sn_xyz, node_size=25,
                     display_mode = 'ortho',
                     colorbar=False, edge_cmap=cm,
                     edge_norm = norm,
                     node_color = 'black',
                     node_kwargs={'alpha':0.5},
                     edge_kwargs={'linewidth': 5},
                     axes=ax)
        if path is not None:
            plt.savefig(path+trait_name.replace(' ','')+'corplot_ctm.png',
                        dpi = 300,
                        bbox_inches = 'tight',
                        facecolor='white')
        plt.show()
    if plt_ctm_sq:
        fig1 = plt.figure(constrained_layout=False)
        gs = fig1.add_gridspec(2, 2, wspace = 0.05, left=0.05, right=0.48)
        ax1 = fig1.add_subplot(gs[0, :])
        plot_connectome(A, node_coords=sn_xyz, node_size=25,
                     display_mode = 'x',
                     colorbar=False, edge_cmap=cm,
                     edge_norm = norm,
                     node_color = 'black',
                     node_kwargs={'alpha':0.5},
                     edge_kwargs={'linewidth': 5},
                     axes=ax1)
        ax2 = fig1.add_subplot(gs[1, 0])
        plot_connectome(A, node_coords=sn_xyz, node_size=25,
                     display_mode = 'y',
                     colorbar=False, edge_cmap=cm,
                     edge_norm = norm,
                     node_color = 'black',
                     node_kwargs={'alpha':0.5},
                     edge_kwargs={'linewidth': 5},
                     axes=ax2)
        ax3 = fig1.add_subplot(gs[1, 1])
        plot_connectome(A, node_coords=sn_xyz, node_size=25,
                     display_mode = 'z',
                     colorbar=False, edge_cmap=cm,
                     edge_norm = norm,
                     node_color = 'black',
                     node_kwargs={'alpha':0.5},
                     edge_kwargs={'linewidth': 5},
                     axes=ax3)
        if path is not None:
            plt.savefig(path+trait_name.replace(' ','')+'corplot_ctm.png',
                        dpi = 300,
                        bbox_inches = 'tight',
                        facecolor='white')
        plt.show()
    return

def ctm_plot_hk_modulation(bnd, xyz, path=None, make_colorbar=True):
    """Method for creating plots for results from heat kernel modulation analyses as in Thomas et al., 2021. Note that because this creates on set of brain plots, it only works when hk_modulation() is run with group_min=True.

    Parameters
    ----------
    bnd : brainnetdiff object
        brainnetdiff object
    xyz : numpy array
        array of shape (N, 3) containing the xyz coordinates corresponding to rois supplied in the parameter roi_names of the brainnetdiff object.
    path : string
        filepath+name for saving plots. If None (default) plots will not be saved.
    make_colorbar : boolean
        True (default) will also generate colorbar/legend for plot

    Returns
    -------
    self : returns an instance of self
    """
    if not bnd.group_min:
        raise ValueError('Must run hk_modulation() method with group_min=True for this plotting method')
    if bnd.by_region:
        sn_xyz, sn_names = bnd.avg_reg_xyz(xyz)
    else:
        sn_xyz, sn_names = bnd.get_sn_xyz(xyz)
    roi_name = bnd.get_opt_modulation_df().node.values[0]
    roi_ind = bnd.get_opt_modulation_df().node_idx.values[0]
    roi_ind = roi_ind.astype(int)
    nsize = np.array([100]*len(sn_names))
    nsize[roi_ind] = 300
    ncolor = ['gray']*len(sn_names)
    ncolor[roi_ind] = 'black'
    val_map0 = {(1,0):1,
               (1,1):1,
               (1,-1):1}
    val_map1 = {(1,0):1, #elim
               (1,1):2, #remain
               (0,-1):3, #new
               (1,-1):4} #overcomp
    A1 = bnd.make_adj_mat(val_map0)
    A1 = A1 + A1.T
    A2 = bnd.make_adj_mat(val_map1)
    A2 = A2 + A2.T
    colors = [(0,0,0), (0,0,1)]
    cm1 = LinearSegmentedColormap.from_list(
            'my_cm', colors, N=1)
    cm2 = mpl.colors.ListedColormap(['lightblue', 'blue', 'tomato', 'crimson'])
    bounds = [0, 1.1, 2.1, 3.1, 4.1]
    norm = mpl.colors.BoundaryNorm(bounds, cm2.N)
    fig, axes = plt.subplots(nrows=2, figsize=(15,10))
    plot_connectome(A1, node_coords=sn_xyz, node_size=nsize,
                     colorbar=False, edge_cmap=cm1,
                     edge_norm=None,
                     node_color = ncolor,
                     node_kwargs={'alpha':0.75},
                     edge_kwargs={'linewidth':4},
                     title = None,
                     axes=axes[0])
    plot_connectome(A2, node_coords=sn_xyz, node_size=nsize,
                     colorbar=False, edge_cmap=cm2,
                     edge_norm=norm,
                     node_color = ncolor,
                     node_kwargs={'alpha':0.75},
                     edge_kwargs={'linewidth':4},
                     title = None,
                     axes=axes[1])
    fd = {'fontsize':16, 'fontweight':'bold'}
    axes[0].set_title('Baseline', loc='left', fontdict=fd)
    axes[1].set_title('Additional heat at '+roi_name, loc='left',fontdict=fd)
    if path is not None:
        fig.savefig(path+roi_name.replace(' ','_').replace('/','-')+'ctm_plot_addheat.png',
                    dpi = 300,
                    bbox_inches = 'tight',
                    facecolor='white')
    plt.show()
    if make_colorbar:
        custom_lines = [Line2D([0], [0], color=cm2(0), lw=4,
                        label='corrected'),
                        Line2D([0], [0], color=cm2(1), lw=4,
                        label='decreased at baseline'),
                        Line2D([0], [0], color=cm2(2), lw=4,
                        label='new increase (no diff at baseline)'),
                        Line2D([0], [0], color=cm2(3), lw=4,
                        label='new (diff at baseline opposite dir)')]
        fig, ax = plt.subplots()
        ax.legend(handles=custom_lines, loc='center')
        ax.axis('off')
        if path is not None:
            fig.savefig(path+roi_name.replace(' ','_').replace('/','-')+'addheat_legend.png',
                        dpi = 300,
                        bbox_inches = 'tight',
                        facecolor='white')
        plt.show()
    return

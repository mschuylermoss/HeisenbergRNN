import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors


scales = [0.25,0.5,1.,2.,4.]
rates = [0.25,0.475]

colorscale = plt.get_cmap('plasma')
shades = np.linspace(0.,0.9,len(scales)+1)[::-1][:-1]
colors = colorscale(shades)
markers = {0.475:'s',0.25:'8'}
markersizes = {0.475:6,0.25:8}
alphas = {0.475:0.5,0.25:1.0}
linestyles= {0.475:'--',0.25:'-'}
bigger_font_size = 22

colorscale_C = plt.get_cmap('viridis')
colors_C = colorscale_C(shades)

Ls_1 = np.arange(6,21,2)
Ls_2 = np.arange(24,33,4)
Ls_all = np.concatenate((Ls_1,Ls_2),axis=0)

def add_energy_legends(reference_label=None,show_runs=False,show_zero_var=False,show_C=False):
    if show_runs:
        runs_markers = [
        plt.Line2D([0], [0], marker=markers[rate], linestyle='None', color='grey', alpha=alphas[rate], markersize=markersizes[rate], label=f'rate $r$={rate}') for rate in rates
        ]
    else:
        runs_markers = []
    if show_zero_var: 
        zero_var_markers = [
        plt.Line2D([0], [0], marker='o', linestyle='-',color='k', markeredgecolor='k', markerfacecolor='white', markersize=8, label=f'RNN zero variance'),
        ]
    else:
        zero_var_markers = []
    if reference_label is not None:
        reference_markers = [
        plt.Line2D([0], [0], marker='*', linestyle='None',  color='k',markersize=8, label=f'{reference_label}'),
        plt.Line2D([0], [0], linestyle='--', color='k', label=f'{reference_label} TL')
        ]
    else:
        reference_markers = []
    plt.legend(handles=runs_markers+zero_var_markers+reference_markers, loc='best')
    
    if show_runs:
        bounds = [0, 1, 2, 3, 4, 5]  # Boundaries for discrete bins
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
        sm.set_array([]) 
        if show_C:
            pad_=-0.1
        else:
            pad_=0.02
        cbar = plt.colorbar(sm, ticks=[0.5, 1.5, 2.5, 3.5, 4.5], aspect=30, pad=pad_)
        cbar.ax.set_yticklabels(['0.25', '0.50', '1.0', '2.0', '4.0'])  # Custom labels
        cbar.set_label("Scale $s$",fontsize=15)
        cbar.ax.xaxis.set_label_position('top')  # Position the label at the top
        # cbar.ax.set_position([0.9, 0.1, 0.05, 0.5])  # [left, bottom, width, height]
        # cbar.ax.set_xlabel("Scale",labelpad=5)  # Set the horizontal label explicitly
        if show_C:
            cmap_C = mcolors.ListedColormap(colors_C)
            norm_C = mcolors.BoundaryNorm(bounds, cmap_C.N)
            sm_C = plt.cm.ScalarMappable(cmap=cmap_C,norm=norm_C)
            cbar_C = plt.colorbar(sm_C, ticks=[0.5, 1.5, 2.5, 3.5, 4.5], aspect=30, pad=0.02)
            cbar_C.ax.set_yticklabels([])  # Custom labels
            cbar_C.ax.tick_params(length=0)

def add_legends_Sk(which_scale_i=0,reference_label=None):
    custom_markers_Sk = [
    plt.Line2D([0], [0], marker=markers[rate], linestyle='None', color=colors[which_scale_i], alpha=alphas[rate], markersize=markersizes[rate], label=f'$M^{2}$, rate  $r$={rate}') for rate in rates
    ]
    custom_markers_C = [
    plt.Line2D([0], [0], marker=markers[rate], linestyle='None', color=colors_C[which_scale_i], alpha=alphas[rate], markersize=markersizes[rate], label=f'$M^{2}_C$, rate $r$={rate}') for rate in rates
    ]
    if reference_label is not None:
        reference_markers = [
        plt.Line2D([0], [0], marker='*', linestyle='None',  color='k',markersize=8, label=f'{reference_label}'),
        plt.Line2D([0], [0], linestyle='--', color='k', label=f'{reference_label} TL')
        ]
    else:
        reference_markers = []
    plt.legend(handles=custom_markers_Sk + custom_markers_C +reference_markers, loc='best')

def add_gridlines(axes='both'):
    if axes=='both':
        plt.grid(visible=True, which='major', linestyle='-', linewidth=0.75, alpha=0.9)  # Major gridlines
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)  # Minor gridlines
    else:
        plt.grid(visible=True, which='major', linestyle='-', linewidth=0.75, alpha=0.9, axis=axes)  # Major gridlines
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7, axis=axes)  # Minor gridlines
    plt.gca().set_axisbelow(True)


def add_runtime_legends():
    runs_markers = [
    plt.Line2D([0], [0], marker=markers[rate], linestyle='None', color='grey', alpha=alphas[rate], markersize=markersizes[rate], label=f'rate $r$={rate}') for rate in rates
    ]
    const_markers = [
    plt.Line2D([0], [0], linestyle='-', color='grey', label=f'constant')
    ]
    plt.legend(handles=runs_markers+const_markers, loc='best')
    
    bounds = [0, 1, 2, 3, 4, 5]  # Boundaries for discrete bins
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([]) 
    cbar = plt.colorbar(sm, ticks=[0.5, 1.5, 2.5, 3.5, 4.5], aspect=30, pad=0.03)
    cbar.ax.set_yticklabels(['0.25', '0.50', '1.0', '2.0', '4.0'])  # Custom labels
    cbar.set_label("Scale $s$",fontsize=15)
    cbar.ax.xaxis.set_label_position('top')  # Position the label at the top

def set_x_ticks(axes,cutoff=0):
    if cutoff != 0:
        Ls = Ls_all[:cutoff]
    else:
        Ls = Ls_all
    axes.set_xticks(Ls)
    axes.set_xticks(np.arange(min(Ls),max(Ls)+1,1),minor=True)

import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

scales = [1.,2.,4.]
rates = [0.158,0.25,0.475]
Ls_all = np.arange(6,31,6)

rate_markers={0.158:'*',0.25:'H',0.475:'^'}
rate_lines={0.158:':',0.25:'--',0.475:'-'}
rate_alphas={0.475:0.2,0.25:0.5,0.158:0.9}
rate_markersize={0.158:9,0.25:7,0.475:7}

colorscale_sq = plt.get_cmap('plasma')
colorscale_tri = plt.get_cmap('viridis')
shades = np.linspace(0.,0.9,2*len(scales))[::-2]
colors_sq = colorscale_sq(shades)
colors_tri = colorscale_tri(shades)
scale_shades = {'Square':{1.0:colors_sq[-3],2.0:colors_sq[-2],4.0:colors_sq[-1]},
                'Triangular':{1.0:colors_tri[-3],2.0:colors_tri[-2],4.0:colors_tri[-1]}}
scale_shades_lists = {'Square':colors_sq[0:4],
                      'Triangular':colors_tri[0:4]}

def add_energy_legends(reference_label=None,show_runs=False,show_zero_var=False,both_MS=False,show_const=False):

    if show_runs:
        runs_markers = [
        plt.Line2D([0], [0], marker=rate_markers[rate], linestyle=rate_lines[rate], color='grey', alpha=rate_alphas[rate], markersize=rate_markersize[rate], label=f'rate $r$ = {rate}') for rate in rates
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
        plt.Line2D([0], [0], linestyle='--', color='k', label=f'{reference_label} TL')
        ]
    else:
        reference_markers = []
    
    handles = runs_markers+zero_var_markers+reference_markers
    if show_const: 
        handles += [
            plt.Line2D([0], [0], marker='None', linestyle='-', color='grey', label=f'constant') 
        ]
    plt.legend(handles=handles, loc='best')
    
    if show_runs:
        bounds = [0, 1, 2, 3]  # Boundaries for discrete bins
        cmap = mcolors.ListedColormap(scale_shades_lists['Triangular'])
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
        sm.set_array([]) 
        if both_MS:
            pad_=-0.1
        else:
            pad_=0.02
        cbar = plt.colorbar(sm, ticks=[0.5, 1.5, 2.5], aspect=30, pad=pad_)
        cbar.ax.set_yticklabels(['1.0', '2.0', '4.0'])  # Custom labels
        cbar.set_label("Scale",fontsize=15)
        cbar.ax.xaxis.set_label_position('top')  
        if both_MS:
            cmap_C = mcolors.ListedColormap(scale_shades_lists['Square'])
            norm_C = mcolors.BoundaryNorm(bounds, cmap_C.N)
            sm_C = plt.cm.ScalarMappable(cmap=cmap_C,norm=norm_C)
            cbar_C = plt.colorbar(sm_C, ticks=[0.5, 1.5, 2.5, 3.5], aspect=30, pad=0.02)
            cbar_C.ax.set_yticklabels([])  # Custom labels
            cbar_C.ax.tick_params(length=0)

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


def set_x_ticks(axes,cutoff=0):
    if cutoff != 0:
        Ls = Ls_all[:cutoff]
    else:
        Ls = Ls_all
    axes.set_xticks(Ls)
    axes.set_xticks(np.arange(min(Ls),max(Ls)+1,1),minor=True)

def highlight_k_points(axs,marker_size):
    axs.scatter(-4*np.pi/3,0,marker='h',edgecolor='red',facecolor='None',s=marker_size)
    axs.scatter(4*np.pi/3,0,marker='h',edgecolor='red',facecolor='None',s=marker_size)
    axs.scatter(-2*np.pi/3,2*np.pi/np.sqrt(3),marker='h',edgecolor='red',facecolor='None',s=marker_size)
    axs.scatter(-2*np.pi/3,-2*np.pi/np.sqrt(3),marker='h',edgecolor='red',facecolor='None',s=marker_size)
    axs.scatter(2*np.pi/3,-2*np.pi/np.sqrt(3),marker='h',edgecolor='red',facecolor='None',s=marker_size)
    axs.scatter(2*np.pi/3,2*np.pi/np.sqrt(3),marker='h',edgecolor='red',facecolor='None',s=marker_size)


def get_new_cmap(color):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(color[0], 1, N)[::-1]
    vals[:, 1] = np.linspace(color[1], 1, N)[::-1]
    vals[:, 2] = np.linspace(color[2], 1, N)[::-1]
    newcmp = ListedColormap(vals)
    return newcmp
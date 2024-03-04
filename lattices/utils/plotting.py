import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
import networkx as nx
import numpy.typing as npt
from scipy.interpolate import griddata
from matplotlib import cm
import os
import plotly.express as px
import plotly.graph_objects as go
from math import floor
from typing import Optional, Iterable, Tuple


def parity_plot(true, pred, fn: str):
    fig, ax = plt.subplots()
    ax.scatter(true, pred, s=15, alpha=0.5)
    ax.axline((true.min(),true.min()),slope=1,c='k')
    ax.set_xlabel(f'True')
    ax.set_ylabel(f'Predicted')
    validAREL = np.mean(np.abs(pred-true)/true)
    ax.text(0.95,0.05,
        f'MAPE {100*validAREL:.1f}%',
        fontsize=12,
        transform=ax.transAxes,
        horizontalalignment='right') 
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn, dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.1)

def surface_plot(true, pred, dataset, nplot: int, fn: str):
    fig = plt.figure(figsize=(6,3*nplot))
    names = np.array(dataset.data.name)
    uq_names = np.unique(names)
    nplot = min(nplot, len(uq_names))
    plot_uq_names = np.random.choice(uq_names, nplot, replace=False)
    for i in range(nplot):
        name = plot_uq_names[i]
        idx = np.flatnonzero(names==name)
        phi = [dataset[k].phi.item() for k in idx]
        th = [dataset[k].th.item() for k in idx]
        ax = fig.add_subplot(nplot,2,1+(2*i), projection='3d')
        plot_surface_data(phi, th, true[idx], ax=ax, resolution=400j, clims=(1,2))
        ax = fig.add_subplot(nplot,2,2+(2*i), projection='3d')
        plot_surface_data(phi, th, pred[idx], ax=ax, resolution=400j, clims=(1,2))
    plt.subplot(nplot, 2, 1).set_title('Ground truth')
    plt.subplot(nplot, 2, 2).set_title('Prediction')
    plt.tight_layout()
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    plt.savefig(fn, dpi=300, facecolor='w', bbox_inches='tight', pad_inches=0.1)

def get_nodes_edge_coords(
    lat, repr, coords, highlight_nodes: Optional[Iterable]=None
) -> Tuple[np.ndarray, np.ndarray, Iterable, np.ndarray, np.ndarray]:

    nodes = lat.reduced_node_coordinates

    if coords=='transformed': 
        Q = lat.transform_matrix
    elif coords=='reduced':
        Q = np.eye(3)
    else: 
        raise ValueError
    Q6 = np.block([[Q, np.zeros_like(Q)],[np.zeros_like(Q), Q]])

    if repr=='cropped':
        edges = lat.edge_adjacency
        edge_coords = lat._node_adj_to_ec(nodes, edges)
        node_numbers = np.arange(nodes.shape[0])
        try:
            edge_widths = lat.windowed_edge_radii
        except AttributeError:
            edge_widths = 2*np.ones(edges.shape[0])
    elif repr=='fundamental':
        if not hasattr(lat, 'fundamental_edge_adjacency'):
            lat.calculate_fundamental_representation()
        edges = lat.fundamental_edge_adjacency
        edge_coords = lat._node_adj_to_ec(nodes, edges)
        edge_coords += lat.fundamental_tesselation_vecs
        uq_inds = np.unique(edges)
        nodes = nodes[uq_inds] # only plot the fundamental nodes
        if isinstance(highlight_nodes, Iterable):
            assert np.all(np.in1d(highlight_nodes, uq_inds)), \
                "Highlighted node must be a fundamental node"
            highlight_nodes = np.searchsorted(uq_inds, highlight_nodes)
        node_numbers = uq_inds
        try:
            edge_widths = lat.fundamental_edge_radii
        except AttributeError:
            edge_widths = 2*np.ones(edges.shape[0])
    else:
        raise ValueError
    
    # scale mean to 2
    edge_widths = 2*edge_widths/np.mean(edge_widths)

    return nodes@(Q.T), edge_coords@(Q6.T), highlight_nodes, node_numbers, edge_widths

def plot_unit_cell_2d(
    lat, repr='cropped', coords='reduced', show_node_numbers=True,
    ax=None
    ) -> plt.Axes:
    
    nodes, edge_coords, _, node_numbers, edge_widths = get_nodes_edge_coords(
        lat, repr, coords
    )
    
    if not isinstance(ax, plt.Axes):
        fig = plt.figure(figsize=(5,5),facecolor='w')
        ax = plt.axes()
    ax.scatter(nodes[:,0], nodes[:,1])
    segments = []
    colors = [] 
    for i_e, e in enumerate(edge_coords):
        p0, p1 = e[:3], e[3:]
        x_0, y_0, z_0 = p0
        x_1, y_1, z_1 = p1
        segments.append([(x_0, y_0), 
                        (x_1, y_1)])
        colors.append(f'C{i_e%5}')

    lc = LineCollection(segments, colors=colors, linewidths=edge_widths)
    ax.add_collection(lc)
    if show_node_numbers:
        for n, num in zip(nodes, node_numbers):
            ax.text(n[0],n[1],f"{num}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax

def plot_unit_cell_3d(
    lat, repr='cropped', coords='reduced', show_node_numbers=False,
    show_uc_box: bool = False,
    ax=None
    ) -> plt.Axes:

    nodes, edge_coords, _, node_numbers, edge_widths = get_nodes_edge_coords(
        lat, repr, coords
    )
    
    if not isinstance(ax, plt.Axes):
        fig = plt.figure(figsize=(5,5),facecolor='w')
        ax = plt.axes(projection='3d')

    if show_uc_box:
        pts = np.array(
            [
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [0,1,0],
                [0,0,1],
                [1,0,1],
                [1,1,1],
                [0,1,1]
            ]
        )
        if coords=='transformed':
            pts = lat.transform_coordinates(pts)
        inds = [1,0,3,2,None,0,4,7,3,None,4,5,6,7,None,5,1,2,6]
        segments = []
        for i0, i1 in zip(inds[:-1], inds[1:]):
            if i0 is not None and i1 is not None:
                segments.append([pts[i0], pts[i1]])
        uc_box = Line3DCollection(segments, colors='black', linewidths=1)
        ax.add_collection(uc_box)      

    ax.scatter(nodes[:,0], nodes[:,1], nodes[:,2])
    segments = []
    colors = [] 
    for i_e, e in enumerate(edge_coords):
        p0, p1 = e[:3], e[3:]
        x_0, y_0, z_0 = p0
        x_1, y_1, z_1 = p1
        segments.append([(x_0, y_0, z_0), 
                        (x_1, y_1, z_1)])
        colors.append(f'C{i_e%5}')
    lc = Line3DCollection(segments, colors=colors, linewidths=edge_widths)
    ax.add_collection(lc)
    if show_node_numbers:
        for n, num in zip(nodes, node_numbers):
            ax.text(n[0],n[1],n[2],f"{num}")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax

def plotly_unit_cell_3d(
    lat, repr='cropped', coords='reduced', show_node_numbers=False, 
    fig=None, subplot: Optional[dict] = None,
    highlight_nodes: Optional[Iterable] = None,
    highlight_edges: Optional[Iterable] = None,
    show_uc_box: bool = False
    ):
    nodes, edge_coords, highlight_nodes, node_numbers, edge_widths = get_nodes_edge_coords(
        lat, repr, coords, highlight_nodes
    )

    colororder = px.colors.qualitative.G10

    x,y,z = nodes.T
    if isinstance(highlight_nodes, Iterable):
        assert np.all(np.in1d(highlight_nodes, np.arange(nodes.shape[0]))), \
            "Highlighted nodes outside of limits"
        colors = ['rgba(40,40,40,0.3)' for _ in range(len(x))]
        for i_node_highlight in highlight_nodes:
            colors[i_node_highlight] = 'rgb(255,0,0)'
    else:
        colors = [colororder[i%10] for i in range(len(x))]
    if isinstance(highlight_edges, Iterable):
        assert np.min(highlight_edges)>=0 and np.max(highlight_edges)<len(edge_coords), \
            "Highlighted edges outside of limits"
        edge_colors = ['rgba(40,40,40,0.3)' for _ in range(len(edge_coords))]
        for i_edge_highlight in highlight_edges:
            edge_colors[i_edge_highlight] = 'rgb(255,0,0)'
    else:
        edge_colors = [colororder[i%10] for i in range(len(edge_coords))]

    if not isinstance(fig, go.Figure):
        fig = go.Figure()
    mode = 'text+markers' if show_node_numbers else 'markers'
    if isinstance(subplot, dict):
        subplot_args = dict(
                row=floor(subplot['index']/subplot['ncols']) + 1,
                col=subplot['index']%subplot['ncols'] + 1
        )
    else:
        subplot_args = {}

    if show_uc_box:
        pts = np.array(
            [
                [0,0,0],
                [1,0,0],
                [1,1,0],
                [0,1,0],
                [0,0,1],
                [1,0,1],
                [1,1,1],
                [0,1,1]
            ]
        )
        if coords=='transformed':
            pts = lat.transform_coordinates(pts)
        inds = [1,0,3,2,None,0,4,7,3,None,4,5,6,7,None,5,1,2,6]

        fig.add_scatter3d(
            x=[pts[i,0] if isinstance(i,int) else None for i in inds],
            y=[pts[i,1] if isinstance(i,int) else None for i in inds],
            z=[pts[i,2] if isinstance(i,int) else None for i in inds],
            mode='lines',
            line=dict(color='black', width=2),
            name='unit cell',
            showlegend=False,
            **subplot_args
        )                    

    fig.add_scatter3d(
        x=x, y=y, z=z,
        marker={'color':colors,'size':6},
        mode=mode,
        text=node_numbers,
        textfont={'size':14},
        showlegend=False,
        name='nodes',
        **subplot_args
    )
    colors = [] 
    x = []
    y = []
    z = []
    for i_e, e in enumerate(edge_coords):
        n0, n1 = e[:3], e[3:]
        x_0, y_0, z_0 = n0
        x_1, y_1, z_1 = n1
        x.extend([x_0, x_1, None])
        y.extend([y_0, y_1, None])
        z.extend([z_0, z_1, None])
        col = edge_colors[i_e]
        colors.extend([col, col, col])

    fig.add_scatter3d(
        x=x, y=y, z=z,
        line={'width':7,'color':colors},
        mode='lines',
        name='edges',
        hoverinfo='none',
        connectgaps=False,
        showlegend=False,   
        **subplot_args
    )
    if hasattr(lat, 'name'):
        title = lat.name
    else:
        title = ''
    if isinstance(subplot, dict):
        fig.layout.annotations[subplot['index']].update(text=title)
    else:
        fig.update_layout(title=title)
    return fig

# %%
def visualize_graph(
    edges : npt.NDArray, nodes= None,  
    node_types=None, ax=None
    ) -> plt.Axes:
    if not isinstance(nodes, np.ndarray):
        nodes = np.unique(edges)
    colors = {'corner':'blue', 'edge':'red', 'face':'green', 'inside':'grey'}
    cmap = ['grey' for i in range(len(nodes))]
    if isinstance(node_types, dict):
        for ntype in node_types.keys():
            for n in node_types[ntype]:
                i = n
                cmap[i] = colors[ntype]
    edges_in = []
    edges_count = []
    for e in edges:
        e_sorted = sorted(e)
        if e_sorted in edges_in:
            edges_count[edges_in.index(e_sorted)] += 1
        else:
            edges_in.append(e_sorted)
            edges_count.append(1)
    edges_tuples = []
    for e_sorted in edges_in:
        for i in range(edges_count[edges_in.index(e_sorted)]):
            e = list(e_sorted)
            edges_tuples.append((e[0],e[1],{'r':i}))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges_tuples)
    #
    if isinstance(ax, plt.Axes):
        pass
    else:
        fig = plt.figure(facecolor='w')
        ax = plt.gca()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos,  
        node_color = cmap, node_size = 200, alpha = 1, ax=ax
    )
    nx.draw_networkx_labels(G, pos, ax=ax)

    edges_in = []
    edges_count = []
    for j,e in enumerate(edges_tuples):
        ec="0"
        if e[0]==e[1]:
            c = [x+0.1 for x in pos[e[0]]]
            circ = patches.Circle(xy=c, radius=0.1/np.sqrt(2), fc='none', ec='0')
            ax.add_patch(circ)
        else:
            if len(e)>2: r=0.3*e[2]['r']
            else: r=0
            ax.annotate("",
                        xy=pos[e[0]], xycoords='data',
                        xytext=pos[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="-", color=ec,
                                        shrinkA=10, shrinkB=10,
                                        patchA=None, patchB=None,
                                        connectionstyle=f"arc3,rad=rrr".replace('rrr',str(r)),
                                        ),
                        )
    plt.axis('off')
    return ax
# %%
def plotly_elasticity_surf(
    S: np.ndarray, 
    title: str='',
    fig=None, subplot: Optional[dict] = None,
    clim: Optional[Tuple[float, float]] = None,
    ):
    assert S.shape==(3,3,3,3)
        
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]


    X = np.sin(v)*np.cos(u)
    Y = np.sin(v)*np.sin(u)
    Z = np.cos(v)

    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    pos = np.column_stack((x,y,z))

    e = 1/np.einsum('ai,aj,ak,al,ijkl->a',pos,pos,pos,pos,S)

    rows, cols = X.shape
    indices = np.unravel_index(np.arange(len(e)), (rows, cols))
    E = np.zeros_like(X)
    E[indices] = e

    R = E

    X = R*np.sin(v)*np.cos(u)
    Y = R*np.sin(v)*np.sin(u)
    Z = R*np.cos(v)

    if not isinstance(fig, go.Figure):
        fig = go.Figure()
    if isinstance(subplot, dict):
        subplot_args = dict(
                row=floor(subplot['index']/subplot['ncols']) + 1,
                col=subplot['index']%subplot['ncols'] + 1
        )
    else:
        subplot_args = {}

    clim = clim or (np.min(R), np.max(R))
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z, surfacecolor=R, cmin=clim[0], cmax=clim[1]),
        **subplot_args
    )

    if isinstance(subplot, dict):
        fig.layout.annotations[subplot['index']].update(text=title)
    else:
        fig.update_layout(title=title)
    return fig

def plotly_stiffness_surf(
    C: np.ndarray, 
    title: str='',
    fig=None, subplot: Optional[dict] = None,
    clim: Optional[Tuple[float, float]] = None,
    resolution: int = 100j,
    ) -> go.Figure:
    assert C.shape==(3,3,3,3)
        
    u, v = np.mgrid[0:2*np.pi:resolution, 0:np.pi:resolution]


    X = np.sin(v)*np.cos(u)
    Y = np.sin(v)*np.sin(u)
    Z = np.cos(v)

    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    pos = np.column_stack((x,y,z))

    e = np.einsum('ai,aj,ak,al,ijkl->a',pos,pos,pos,pos,C)

    rows, cols = X.shape
    indices = np.unravel_index(np.arange(len(e)), (rows, cols))
    E = np.zeros_like(X)
    E[indices] = e

    R = E

    X = R*np.sin(v)*np.cos(u)
    Y = R*np.sin(v)*np.sin(u)
    Z = R*np.cos(v)

    if not isinstance(fig, go.Figure):
        fig = go.Figure()
    if isinstance(subplot, dict):
        subplot_args = dict(
                row=floor(subplot['index']/subplot['ncols']) + 1,
                col=subplot['index']%subplot['ncols'] + 1
        )
    else:
        subplot_args = {}

    clim = clim or (np.min(R), np.max(R))
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z, surfacecolor=R, cmin=clim[0], cmax=clim[1], lighting=dict(roughness=0.25, specular=0.1)),
        **subplot_args
    )

    if isinstance(subplot, dict):
        fig.layout.annotations[subplot['index']].update(text=title)
    else:
        fig.update_layout(title=title)
    return fig

def plotly_scaling_surf(
    C: np.ndarray, 
    rel_dens: Iterable[float],
    title: str='',
    fig: Optional[go.Figure] = None, 
    subplot: Optional[dict] = None,
    resolution: int = 100j,
    ) -> go.Figure:
    """Plot the surface of scaling exponent for a given stiffness tensor

    Args:
        C (np.ndarray): stacked compliance tensors [n_rel_dens, 3,3,3,3]
        rel_dens (Iterable[float]): corresponding relative densities
        title (str, optional): title for plot. Defaults to ''.
        fig (Optional[dict], go.Figure): can pass an existing figure. Defaults to None.
        subplot (Optional[dict], optional): subplot arguments. Expect keys ['index','ncol']. Defaults to None.

    Returns:
        go.Figure: plotly figure
    """

    #     C (np.ndarray): stacked compliance tensors [n_rel_dens, 3,3,3,3]
    assert C.ndim==5
    assert C.shape[1:]==(3,3,3,3)
    assert len(rel_dens)==C.shape[0]
        
    u, v = np.mgrid[0:2*np.pi:resolution, 0:np.pi:resolution]


    X = np.sin(v)*np.cos(u)
    Y = np.sin(v)*np.sin(u)
    Z = np.cos(v)

    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    pos = np.column_stack((x,y,z))

    e = 1/np.einsum('pi,pj,pk,pl,...ijkl->...p',pos,pos,pos,pos,C) # [rel_dens, direction]
    # linear fit
    x_fit = np.log(rel_dens)
    y_fit = np.log(e)
    fit = np.polyfit(x_fit, y_fit, 1)
    n = fit[0] # [direction]

    rows, cols = X.shape
    indices = np.unravel_index(np.arange(len(n)), (rows, cols))
    N = np.zeros_like(X)
    N[indices] = n

    R = N

    X = R*np.sin(v)*np.cos(u)
    Y = R*np.sin(v)*np.sin(u)
    Z = R*np.cos(v)

    if not isinstance(fig, go.Figure):
        fig = go.Figure()
    if isinstance(subplot, dict):
        subplot_args = dict(
                row=floor(subplot['index']/subplot['ncols']) + 1,
                col=subplot['index']%subplot['ncols'] + 1
        )
    else:
        subplot_args = {}

    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z, surfacecolor=R, cmin=1, cmax=2),
        **subplot_args
    )

    if isinstance(subplot, dict):
        fig.layout.annotations[subplot['index']].update(text=title)
    else:
        fig.update_layout(title=title)
    return fig
# %%
def plot_surface_data(phi, th, val, ax=None, resolution=200j, clims=None):
    PHI, TH = np.mgrid[0:np.pi:resolution, 0:2*np.pi:resolution]
    R = griddata(np.column_stack((phi,th)), val, (PHI,TH))
    if not isinstance(ax, plt.Axes):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    z = R*np.cos(PHI)
    x = R*np.sin(PHI)*np.cos(TH)
    y = R*np.sin(PHI)*np.sin(TH)
    if isinstance(clims, tuple):
        color_val = (R-clims[0])/(clims[1]-clims[0])
        maxlim = 1.1*clims[1]
        ax.set_xlim(-maxlim, maxlim)
        ax.set_ylim(-maxlim, maxlim)
        ax.set_zlim(-maxlim, maxlim)
    else:
        color_val = (R-np.nanmin(R))/(np.nanmax(R)-np.nanmin(R))
    ax.plot_surface(x,y,z, facecolors=cm.viridis(color_val))
    return ax
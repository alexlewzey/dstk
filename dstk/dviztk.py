"""
Data Visualisation toolkit
--------------------------

data vis library mainly wrapper around common plotting libraries, plotly, seaborn


EXAMPLES
--------

histogram with color
--------------------

fig = px.histogram(df, 'value', color='variable', barmode='overlay', opacity=0.7)


plotly geo scatter
------------------

fig = px.scatter_geo(stores_targ, lat='LAT', lon='LONG', hover_name='NAME', color='index1', hover_data=hd,
                     title='cash strapped stores', size='point1')
fig.update_layout(geo_scope='europe')
plot(fig)


plotly bar
----------

fig = px.bar(df, x=, y=, color=)
fig.update_layout({'yaxis': {'tickformat': '.1%', 'title': 'title'}}, yaxis_type='category')


plotly scatter
--------------

hd = {col1: ‘:,’, col2: ‘:.2%’, col3: True}
fig = px.scatter_3d(df, x=, y=, z=, color=, hover_name=, hover_data=hd)
fig.update_traces(marker=dict(size=3))
plot(fig, filename=path.as_posix())


plotly line
-----------

px.line(df, x=, y=, color=, line_shape=’spline’)
fig.update_traces(line=dict(size=3))
fig.update_xaxes(rangeslider_visible=True)

facet line
----------

fig = px.line(df, y='value', x='variable', color='desc', facet_row='level_0', **kwargs)
fig.update_traces(mode='lines+markers')
fig.update_xaxes(matches=None, showticklabels=True)
fig.update_yaxes(matches=None, showticklabels=True)


combine two figures
-------------------

fig1.add_trace(fig2.data[0])

for data in fig2.data:
    fig1.add_trace(data)


seconday axis
-------------


fig1 = px.line(res_now, 'date', 'lb_all')
fig2 = px.line(res_now, 'date', 'calories')

fig = make_subplots(specs=[[{'secondary_y': True}]])
fig.add_trace(fig1.data[0], secondary_y=False)
fig.add_trace(fig2.data[0], secondary_y=True)


axis titles
-----------

fig.update_layout({'xaxis': {'title': 'Cash Strapped Spend Allocation Index'},
                   'yaxis': {'tickformat': '.2%', 'title': f'% of spend in '}})


facet col wrap
--------------

px.line(facet_col=col, facet_col_wrap=5)


category_orders


"""
import itertools
import random
from itertools import cycle

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from pandas._libs.tslibs.timestamps import Timestamp
from plotly.offline import plot
from plotly.subplots import make_subplots
from slibtk import slibtk
from tqdm import tqdm

from dstk.core import *


# colors ###############################################################################################################

def make_rgb() -> str:
    """generate a list of random rgb values of format rgb(0,0,0)"""
    rnum = lambda: random.randint(0, 255)
    return f'rgb({rnum()},{rnum()},{rnum()})'


# my fave colors
default_cat_colors = {'magenta_pink': '#FC0FC0',
                      'electric blue': '#7df9ff',
                      'b': '#adc1c9',
                      'c': '#052d3f',
                      'd': '#6FFFE9',
                      'e': '#316379',
                      'f': '#84a2af'}
colorlst = ['#FC0FC0', '#adc1c9', '#052d3f', '#6FFFE9', '#316379', '#84a2af']
color: str = '#FC0FC0'

rgba = {
    'green main': 'rgba(112, 182, 88, 0.6)',
    'green dark': 'rgba(33, 84, 37, 0.6)',
    'grey dark': 'rgba(49, 45, 49, 0.6)',
    'dog': 'rgba(191, 209, 67, 0.6)',
    'cat': 'rgba(232, 132, 65, 0.6)',
    'small pet': 'rgba(212, 153, 59, 0.6)',
    'fish': 'rgba(40, 58, 140, 0.6)',
    'bird': 'rgba(109, 173, 218, 0.6)',
    'reptile': 'rgba(101, 38, 57, 0.6)',
}
rgba_vals = list(rgba.values())
rgba_inf = cycle(rgba.values())


def plot_histograms(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """plot all varibles in a df as a histogram"""
    cols = columns if columns else df.columns
    for col in cols:
        px.histogram(df, col, title=col).plot()


# plotly ###############################################################################################################


def line_seconday_axis(df: pd.DataFrame, x: str, y_primary: str, y_secondary: str, title: Optional[str] = None,
                       path: Optional[Path] = None) -> None:
    """plot a plotly lines graph with a seconday axis"""
    fig1 = px.line(df, x, y_primary, color_discrete_sequence=['blue'])
    fig2 = px.line(df, x, y_secondary, color_discrete_sequence=['magenta'])
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.update_layout(title=title)
    fig.add_trace(fig1.data[0], secondary_y=False)
    fig.add_trace(fig2.data[0], secondary_y=True)
    fig.update_yaxes(title_text=y_primary, color='blue', secondary_y=False)
    fig.update_yaxes(title_text=y_secondary, color='magenta', secondary_y=True)
    fig.plot(path)


def px_scatter3d_by_colors(df: pd.DataFrame, colors: List, path: Path, fname: str, hover_name: str, hover_data,
                           marker_size: int = 3, title: Optional[str] = None) -> None:
    """plotly a plotly scatter 3d chart for every """
    for color in tqdm(colors):
        fig = px.scatter_3d(df, x='dim0', y='dim1', z='dim2', hover_name=hover_name, color=color,
                            hover_data=hover_data, title=f'{title}{color}')
        fig.update_traces(marker=dict(size=marker_size))
        plot(fig, filename=(path / f'{fname}_{color}.html').as_posix())


def add_vertline(fig: go.Figure, y1: float, x: int = 1) -> go.Figure:
    """add vertical line to at x=1 to plotly figure with height of y1"""
    fig.add_shape(
        type='line',
        x0=x,
        y0=0,
        x1=x,
        y1=y1,
        line=dict(width=5, dash='dot', color='red'),
    )
    return fig


def add_periodic_vertical_lines(fig: go.Figure, start: Union[str, Timestamp], end: Union[str, Timestamp], freq: str,
                                y1: float, y0: float = 0) -> go.Figure:
    """
    add vertical lines to plotly figures that repeat at a certain frequency ie every sunday
    Args:
        fig: plotly figure
        start: first date of period
        end: last date of the period
        freq: pandas time series frequency directive ie W-THU
        y1: max value of line
        y0: minimum value of line

    Returns:
        fig: the plotly figures with lines added as a trace
    """
    dates_dow = pd.date_range(start, end, freq=freq)
    for day in dates_dow:
        fig.add_shape(
            type='line',
            x0=day,
            y0=y0,
            x1=day,
            y1=y1,
            line=dict(width=1, dash='dot'),
        )
    return fig


def make_sankey_fig_from_df(df: pd.DataFrame, source: str = 'source', target: str = 'target',
                            values: str = 'values', title=None,
                            valueformat: str = ',.1%') -> go.Figure:
    """
    Take a correctly formatted DataFrame (cols=[source, target, values] and return a sankey diagram plotly figure object
    Args:
        df: DataFrame with required columns [source, target, values]
        source: column name
        target: column name
        values: column name
        title: chart title string
        valueformat: number formatting of the values in the Sankey diagram

    Returns:
        fig: populated plotly figure object
    """
    sankey_kwargs = _df_to_args_for_sankey(df, source=source, target=target, values=values)
    return make_sankey_fig(valueformat=valueformat, title=title, **sankey_kwargs)


def _df_to_args_for_sankey(df: pd.DataFrame, source: str = 'source', target: str = 'target',
                           values: str = 'values', color_scheme=rgba_inf) -> Dict[str, List]:
    """
    transform df into args required for plotly sankey figure. df must be of format: col_pre=source col_post=destination
    and col_values is the size of the flow between them.
    Args:
        df: DataFrame with required columns [source, target, values]
        source: column name
        target: column name
        values: column name
        color_scheme: An infinite generator of RGB colour strings eg 'rgba(101, 38, 57, 0.6)'

    Returns:
        dict of args required to make sankey diagram with plotly
    """
    index = list(np.concatenate([df[source].unique(), df[target].unique()]))
    source = [index.index(x) for x in df[source]]
    target = [index.index(x) for x in df[target]]
    values = df[values].tolist()
    colors = [next(color_scheme) for _ in range(len(values))]
    return {
        'index': index,
        'source': source,
        'target': target,
        'values': values,
        'colors': colors,
    }


def make_sankey_fig(index: List, source: List, target: List, values: List, colors: List, title: str,
                    valueformat: str = ',.1%') -> go.Figure:
    """
    pass args into plotly figure returning the populated figure, arguments should be passed in as kwargs
    """
    fig = go.Figure(data=[go.Sankey(
        valueformat=valueformat,
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=index,
            color=rgba['green main'],
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=colors
        ))])
    fig.update_layout(title_text=title, font_size=10)
    return fig


def px_scatter_geo(df: pd.DataFrame, color, hover_name, title, path: Optional[Path] = None, lat='latitude',
                   lon='longitude', **kwargs) -> None:
    """convenience wrapper for px.scatter_geo for plotting uk scatter maps"""
    fig = px.scatter_geo(df, lat=lat, lon=lon, hover_name=hover_name, color=color, title=title, **kwargs)
    fig.update_layout(
        geo=dict(
            scope='europe',
            showland=True,
            lonaxis=dict(range=[df[lon].min(), df[lon].max()]),
            lataxis=dict(range=[df[lat].min(), df[lat].max()]),
        ),
    )
    fig.plot(path)


# seaborn ##############################################################################################################


def heatmap(df, width=1.5, height=0.4, cmap=plt.cm.Blues, *args, **kwargs):
    """show a heatmap version of a DataFrame, good for correlation"""
    fig, ax = plt.subplots(1, 1, figsize=(width * df.shape[1], height * df.shape[0]))
    sns.heatmap(df.round(2), annot=True, fmt='g', cmap=cmap, *args, **kwargs, ax=ax)
    plt.tight_layout()
    plt.show()
    return fig, ax


def heatmap_corr(df: pd.DataFrame, thresh=None, width=14, height=8, **kwargs):
    """
    heatmap of principle components
    """
    df = df.corr().round(2)
    if thresh:
        df = df.applymap(lambda x: x if abs(x) > thresh else float('NaN'))
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    sns.heatmap(df, ax=ax, annot=True, **kwargs)
    return (fig, ax)


def violin_and_box(*args, width=12, height=11, **kwargs):
    """
    show a boxplot and a violin plot on a 2x1 subplot grid
    """
    fig, ax = plt.subplots(2, 1, figsize=(width, height))
    sns.violinplot(*args, **kwargs, ax=ax[0])
    sns.boxplot(*args, **kwargs, ax=ax[1])
    plt.show()


# visualising decision boundaries and regression surfaces ##############################################################


def px_scatter3d_regression(df: pd.DataFrame, x: str, y: str, z: str, model, path: Path, title: Optional[str] = None,
                            color_discrete_sequence=None, resolution: int = 200, marker_size: int = 5, *args,
                            **kwargs) -> None:
    """
    plot a 3 dimensional scatter plot where z is the target variable and super impose a regression surface corresponding
    to model where model has a sklearn style api (fit, predict)
    Args:
        df: DataFrame where x and y are features and z is a continuoius label.
        x: column name of feature input 1
        y: column name of feature input 2
        z: column name of continuious label
        model: model object with fit and predict methods ie sklearn
        path: save location of plotly html output
        *args: passed to px.scatter_3d
        **kwargs: passed to px.scatter_3d

    Returns:
        None
    """
    model.fit(df[[x, y]], df[z])
    df['pred'] = model.predict(df[[x, y]])
    surface = make_predicted_surface(df[x], df[y], predictor=model.predict, resolution=resolution)
    # melted = df.melt(id_vars=[x, y], value_vars=[z, 'pred'])
    color_discrete_sequence = color_discrete_sequence if color_discrete_sequence else colorlst
    fig = px.scatter_3d(df, x=x, y=y, z=z, color_discrete_sequence=color_discrete_sequence, title=title, *args,
                        **kwargs)
    fig.update_traces(marker=dict(size=marker_size))
    fig.add_trace(surface)
    plot(fig, filename=path.as_posix())


def make_predicted_surface(x: pd.Series, y: pd.Series, predictor: Callable, resolution: int = 200) -> go.Surface:
    """
    for a given set of values x and y, and a trained model, estimate the grid surface for all permutations
    Args:
        x: first independent variable
        y: Second independent variable
        predictor: Function that is applied to the nx2 array of grid coordinates and returns an nx1 array of predictions
        resolution: number of points on each axis

    Returns:
        surface: plotly surface trace object
    """
    # setting up grid
    x_axis = np.linspace(min(x), max(x), resolution)
    y_axis = np.linspace(min(y), max(y), resolution)
    xx, yy = np.meshgrid(x_axis, y_axis)
    coord = pd.DataFrame(np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]), columns=[x.name, y.name])
    # predicting and formatting
    pred = predictor(coord)
    pred = np.array(pred).reshape(xx.shape)
    surface = {'z': pred.tolist(), 'x': x_axis, 'y': y_axis}
    trace = go.Surface(
        z=surface['z'],
        x=surface['x'],
        y=surface['y'],
        opacity=0.7,
        colorscale=px.colors.sequential.YlGnBu,
        # reversescale=False,
    )
    return trace


def make_3d_grid(df: pd.DataFrame, x: str, y: str, z: str, resolution: int = 50) -> pd.DataFrame:
    """make a DataFrame of dimensional coordinates that uniformly cover a 3 dimensional space"""
    # getting axis ranges
    axis_mins = df[[x, y, z]].min().tolist()
    axis_maxs = df[[x, y, z]].max().tolist()
    # filling axis ranges and combining permutations of all combinations into a DataFrame
    xax = np.linspace(axis_mins[0], axis_maxs[0], resolution)
    yax = np.linspace(axis_mins[1], axis_maxs[1], resolution)
    zax = np.linspace(axis_mins[2], axis_maxs[2], resolution)
    data = list(itertools.product(xax, yax, zax))
    return pd.DataFrame(data, columns=[x, y, z])


def px_scatter3d_classification(df: pd.DataFrame, x: str, y: str, z: str, lbls: str, predictor: Callable, path: Path,
                                resolution: int = 30) -> None:
    """
    plot a 3d set of observations colored by their label with the classification boundary of a predictor superimposed
    on the 3 dimensional plot in the form of equally spaced points that uniformly cover all three dimensions.
    Args:
        df: DataFrame where x and y are features and z is a continuoius label.
        x: column name of feature input 1
        y: column name of feature input 2
        z: column name of feature input 3
        lbls: classification labels
        predictor: Function that is applied to the nx2 array of grid coordinates and returns an nx1 array of predictions
        path: save location of plotly html output
        resolution: number of points on each axis

    Returns:
        None
    """
    color_map = {lbl: make_rgb() for lbl in df[lbls].nunique()}
    fig = px.scatter_3d(df, x, y, z, color='targ',
                        color_discrete_map=color_map)
    grid = make_3d_grid(df, x, y, z, resolution=resolution)
    grid['pred'] = 'cb_' + predictor(grid)
    fig_lbls = px.scatter_3d(grid, x, y, z, color='pred',
                             opacity=0.05,
                             color_discrete_map={f'cb_{k}': v for k, v in color_map.items()})
    for data in fig_lbls.data:
        fig.add_trace(data)
    plot(fig, filename=path.as_posix())


# let reminder of how s**t matplotlib is...


def plt_line(x, y, path='', pct=False, width=9, height=5.5, c='#FC0FC0', **kwargs):
    """
    return a simple line plot
    :return: (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(x, y, color=c, lw=0.9)

    spines = ['left', 'bottom', 'right', 'top']
    [ax.spines[spine].set_visible(False) for spine in spines]
    ax.grid(axis='y', ls='--', alpha=0.6)
    ax.set(**kwargs)

    if pct == True:
        vals = ax.get_yticks()
        ax.set_yticklabels(['{0:.0f}%'.format(x * 100) for x in vals])

    return (fig, ax)


def plt_hbar(df: pd.DataFrame, path='', pct=False, width=9, height=4, sort_index=False, stacked=False,
             color='#052d3f', **kwargs):
    """
    plot a simple horizontal bar chart from a series or DataFrame
    """
    if sort_index:
        df = df.sort_index()
    elif isinstance(df, pd.Series):
        df = df.sort_values()

    fig, ax = plt.subplots(figsize=(width, height))
    df.plot.barh(ax=ax, color=color, stacked=stacked)

    spines = ['top', 'right', 'bottom']
    [ax.spines[spine].set_visible(False) for spine in spines]
    ax.grid(axis='x', ls='--', alpha=0.6)
    ax.set(**kwargs)

    if pct:
        vals = ax.get_xticks()
        ax.set_xticklabels(['{0:.0f}%'.format(x * 100) for x in vals])

    return (fig, ax)


# Monkey patching existing classes #####################################################################################

def _plot(fig: go.Figure, filename: OptPathOrStr = None, auto_open: bool = True, yaxis_pct: bool = False, *args,
          **kwargs) -> None:
    """render plotly figure as html in browser, with default save location as hidden dir in home"""
    (Path.home() / '.plotly').mkdir(exist_ok=True, parents=True)
    # if no filename passed save in a hidden home dir with unique filename so plots always point to separate files
    if not filename: filename = slibtk.next_fname(path=Path.home() / '.plotly', fname='temp-plot', suffix='html')
    filename = Path(filename)
    if yaxis_pct: fig.update_layout({'yaxis': {'tickformat': '.1%'}})
    plot(fig, filename=filename.as_posix(), auto_open=auto_open, *args, **kwargs)


def px_hist(df, col, *args, **kwargs) -> None:
    """monkey patched as a method for DataFrames for easy histograms"""
    fig = px.histogram(df, col, marginal='box', *args, **kwargs)
    fig.plot()


def ser_hist(ser: pd.Series, *args, **kwargs) -> None:
    """Monkey patch as method on panda series for easy histograms"""
    name = ser.name
    fig = px.histogram(ser.to_frame(name), name, marginal='box', *args, **kwargs)
    fig.plot()


def px_barh(df: pd.DataFrame, x: str, y: str, path: Optional[Path] = None) -> None:
    fig = px.bar(df.sort_values(x), y=y, x=x, orientation='h')
    plot(fig, filename=path.as_posix()) if path else plot(fig)


go.Figure.plot = _plot
go.Figure.add_vertline = add_vertline

pd.DataFrame.px_hist = px_hist
pd.DataFrame.px_barh = px_barh

pd.Series.px_hist = ser_hist

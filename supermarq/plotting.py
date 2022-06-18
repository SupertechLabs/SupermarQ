from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from sklearn.linear_model import LinearRegression


def plot_results(
    scores: List[float], tick_labels: List[str], savefn: Optional[str] = None, show: bool = True
) -> None:
    """Plot a simple bar chart of the benchmark results."""
    _, ax = plt.subplots(dpi=150)

    width = 0.4

    xvals = np.arange(len(scores))

    ax.bar(xvals, scores, width=width, tick_label=tick_labels, align="center")

    ax.set_xticks(range(len(scores)))
    ax.set_ylabel("Score")
    plt.tight_layout()

    if savefn is not None:
        # Don't want to save figures when running tests
        plt.savefig(savefn)  # pragma: no cover

    if show:
        # Tests will hang if we show figures during tests
        plt.show()  # pragma: no cover

    plt.close()


def plot_correlations(
    benchmark_features: Dict[str, List[float]],
    device_scores: Dict[str, float],
    feature_labels: List,
    device_name: str,
    savefn: Optional[str] = None,
    show: bool = True,
) -> None:

    temp_correlations = []
    for i in range(len(feature_labels)):
        x, y = [], []
        for benchmark in device_scores.keys():
            x.append(benchmark_features[benchmark][i])
            y.append(device_scores[benchmark])

        X = np.array(x)[:, np.newaxis]
        Y = np.array(y)
        model = LinearRegression().fit(X, Y)
        correlation = model.score(X, Y)
        temp_correlations.append(correlation)

    correlations = np.array([temp_correlations])

    _, ax = plt.subplots(dpi=300)
    im, _ = heatmap(
        correlations,
        [device_name],
        feature_labels,
        ax=ax,
        cmap="cool",
        vmin=0,
        # vmax=0.5,
        cbarlabel=r"Coefficient of Determination, $R^2$",
        cbar_kw={"pad": 0.01},
    )

    annotate_heatmap(im, size=7)

    plt.tight_layout()
    if savefn is not None:
        # Don't want to save figures when running tests
        plt.savefig(savefn)  # pragma: no cover

    if show:
        # Tests will hang if we show figures during tests
        plt.show()  # pragma: no cover
    plt.close()


def plot_benchmark(
    data: List[Union[str, List[str], List[List[float]]]],
    show: bool = True,
    savefn: Optional[str] = None,
    spoke_labels: Optional[List[str]] = None,
    legend_loc: Tuple[float, float] = (0.75, 0.85),
) -> None:
    """
    Create a radar plot of the given benchmarks.

    Input
    -----
    data : Contains the title, feature data, and labels in the format:
        [title, [benchmark labels], [[features_1], [features_2], ...]]
    """
    plt.rcParams["font.family"] = "Times New Roman"

    if spoke_labels is None:
        spoke_labels = ["Connectivity", "Liveness", "Parallelism", "Measurement", "Entanglement"]

    N = len(spoke_labels)
    theta = radar_factory(N)

    _, ax = plt.subplots(dpi=150, subplot_kw=dict(projection="radar"))
    # fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    _, labels, case_data = data
    # ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_rgrids([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
    #             horizontalalignment='center', verticalalignment='center')
    for d, label in zip(case_data, labels):
        ax.plot(theta, d, label=label)
        ax.fill(theta, d, alpha=0.25)
    ax.set_varlabels(spoke_labels)

    ax.legend(loc=legend_loc, labelspacing=0.1, fontsize=11)
    plt.tight_layout()

    if savefn is not None:
        # Don't want to save figures when running tests
        plt.savefig(savefn)  # pragma: no cover

    if show:
        # Tests will hang if we show figures during tests
        plt.show()  # pragma: no cover

    plt.close()


def heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    ax: Optional[Any] = None,
    cbar_kw: Optional[Dict] = None,
    cbarlabel: str = "",
    **kwargs: Any
) -> Tuple[Any, Any]:
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()  # pragma: no cover

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if cbar_kw is None:
        cbar_kw = {}  # pragma: no cover
    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=8)
    cbar.ax.tick_params(labelsize=10)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=8)  # regular fontsize is 12
    ax.set_yticklabels(row_labels, fontsize=7)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate the 3 typical features separately
    for t in ax.get_xticklabels()[-3:]:
        t.set_horizontalalignment("left")
        t.set_rotation(35)

    return im, cbar


def annotate_heatmap(
    im: matplotlib.image,
    data: Optional[np.ndarray] = None,
    valfmt: Any = "{x:.2f}",
    textcolors: Tuple[str, str] = ("black", "white"),
    threshold: Optional[float] = None,
    **textkw: Any
) -> List:
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)  # pragma: no cover
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def radar_factory(num_vars: int) -> np.ndarray:
    """
    (https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html)

    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(RadarAxesMeta):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.frame = "circle"
            self.theta = theta
            self.num_vars = num_vars
            super().__init__(*args, **kwargs)

    register_projection(RadarAxes)
    return theta


class RadarAxesMeta(PolarAxes):

    name = "radar"
    # use 1 line segment to connect specified points
    RESOLUTION = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # rotate plot such that the first axis is at the top
        self.set_theta_zero_location("N")

    def fill(
        self, *args: Any, closed: bool = True, **kwargs: Any
    ) -> List[matplotlib.patches.Polygon]:
        """Override fill so that line is closed by default"""
        return super().fill(closed=closed, *args, **kwargs)

    def plot(self, *args: Any, **kwargs: Any) -> None:
        """Override plot so that line is closed by default"""
        lines = super().plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)

    def _close_line(self, line: matplotlib.lines.Line2D) -> None:
        x, y = line.get_data()
        # FIXME: markers at x[0], y[0] get doubled-up
        if x[0] != x[-1]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            line.set_data(x, y)

    def set_varlabels(self, labels: List[str]) -> None:
        self.set_thetagrids(np.degrees(self.theta), labels, fontsize=14)

    def _gen_axes_patch(self) -> matplotlib.patches.Circle:
        # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
        # in axes coordinates.
        return Circle((0.5, 0.5), 0.5)

    def _gen_axes_spines(self) -> matplotlib.spines.Spine:
        return super()._gen_axes_spines()

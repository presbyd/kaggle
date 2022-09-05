import plotly.graph_objs as go
from plotly.tools import make_subplots
from plotly.offline import iplot, init_notebook_mode
import pandas as pd


def plotFrequency(variable):
    """Plots absolute and relative frequency of a avriable."""

    # Calculates absolute frequency
    absFreq = variable.value_counts()

    # Calculates relative frequency
    relFreq = variable.value_counts(normalize=True).round(4) * 100

    # Creates a dataframe off absolute and relative frequency
    df = pd.DataFrame({"absoluteFrequency": absFreq, "relativeFrequency": relFreq})

    # Create two subplots of bar chart
    fig = make_subplots(
        rows=1,
        cols=2,
        vertical_spacing=0.3,
        subplot_titles=("Absolute Frequency", "Relative Frequency"),
        print_grid=False,
    )  # This suppresses "This is the format of your plot grid:" text from popping out.

    # Add trace for absolute frequency
    fig.add_trace(
        go.Bar(
            y=df.index,
            x=df.absoluteFrequency,
            orientation="h",
            text=df.absoluteFrequency,
            hoverinfo="x+y",
            textposition="auto",
            name="Abs Freq",
            textfont=dict(family="sans serif", size=14),
            marker=dict(color=df.absoluteFrequency, colorscale="Rainbow"),
        ),
        row=1,
        col=1,
    )

    # Add another trace for relative frequency
    fig.add_trace(
        go.Bar(
            y=df.index,
            x=df.relativeFrequency.round(2),
            orientation="h",
            text=df.relativeFrequency.round(2),
            hoverinfo="x+y",
            textposition="auto",
            name="Rel Freq(%)",
            textfont=dict(family="sans serif", size=15),
            marker=dict(color=df.relativeFrequency.round(2), colorscale="Rainbow"),
        ),
        row=1,
        col=2,
    )

    # Update the layout. Add title, dimension, and background color
    fig.layout.update(
        height=600,
        width=970,
        hovermode="closest",
        title_text=f"Absolute and Relative Frequency of {variable.name}",
        showlegend=False,
        paper_bgcolor="rgb(243, 243, 243)",
        plot_bgcolor="rgb(243, 243, 243)",
    )

    # Set y-axis title in bold
    fig.layout.yaxis1.update(title=f"<b>{variable.name}</b>")

    # Set x-axes titles in bold
    fig.layout.xaxis1.update(title="<b>Abs Freq</b>")
    fig.layout.xaxis2.update(title="<b>Rel Freq(%)</b>")
    # or, fig["layout"]["xaxis2"].update(title="<b>Rel Freq(%)</b>")
    return fig.show()


def removeOutliers(variable):
    """Calculates and removes outliers using IQR method."""

    # Calculate 1st, 3rd quartiles and iqr.
    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)
    iqr = q3 - q1

    # Calculate lower fence and upper fence for outliers
    lowerFence, upperFence = (
        q1 - 1.5 * iqr,
        q3 + 1.5 * iqr,
    )  # Any values less than l_fence and greater than u_fence are outliers.

    # Observations that are outliers
    outliers = variable[(variable < lowerFence) | (variable > upperFence)]

    # Drop obsevations that are outliers
    filtered = variable.drop(outliers.index, axis=0).reset_index(drop=True)
    return filtered


def plotBoxPlot(variable, filteredVariable):
    """Plots Box plot of a variable with and without outliers.
    We will also use the output of removeOutliers function as the input to this function.
    variable = variable with outliers,
    filteredVariable = variable without outliers"""

    # Create subplot object.
    fig = make_subplots(
        rows=2,
        cols=1,
        print_grid=False,
        subplot_titles=(
            f"{variable.name} Distribution with Outliers",
            f"{variable.name} Distribution without Outliers",
        ),
    )

    # This trace plots boxplot with outliers
    fig.add_trace(
        go.Box(
            x=variable, name="", marker=dict(color="darkred")  # This removes trace 0
        ),
        row=1,
        col=1,
    )

    # This trace plots boxplot without outliers
    fig.add_trace(
        go.Box(x=filteredVariable, name="", marker=dict(color="green")), row=2, col=1
    )

    # Update layout
    fig.layout.update(
        height=800,
        width=870,
        showlegend=False,
        paper_bgcolor="rgb(243, 243, 243)",
        plot_bgcolor="rgb(243, 243, 243)",
    )

    # Update axes
    fig.layout.xaxis2.update(title=f"<b>{variable.name}</b>")
    return fig.show()


def calculateMissingValues(df):
    dff = df.isna().sum()[
        df.isna().sum() > 0
    ]  # Returns only columns with missing values

    return dff


def plotScatterPlot(x, y, title, yaxis):
    trace = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(color=y, size=35, showscale=True, colorscale="Rainbow"),
    )
    layout = go.Layout(
        hovermode="closest",
        title=title,
        yaxis=dict(title=yaxis),
        height=600,
        width=900,
        showlegend=False,
        paper_bgcolor="rgb(243, 243, 243)",
        plot_bgcolor="rgb(243, 243, 243)",
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig.show()

"""Style surachat / survente du tableau analyse technique."""
import pandas as pd

from analytics import (
    _STYLE_FAVORABLE,
    _STYLE_UNFAVORABLE,
    _technical_signal_cell_style,
    style_technical_table,
)


def test_technical_signal_cell_style():
    assert _technical_signal_cell_style("Surachat") == _STYLE_UNFAVORABLE
    assert _technical_signal_cell_style("Survente") == _STYLE_FAVORABLE
    assert _technical_signal_cell_style("Neutre") == ""
    assert _technical_signal_cell_style("N/A") == ""


def test_style_technical_table_colors_signal_columns():
    df = pd.DataFrame(
        {
            "Ticker": ["AAPL"],
            "Signal RSI": ["Surachat"],
            "Signal Stoch": ["Survente"],
            "Signal MFI": ["Neutre"],
            "RSI (0–100)": [72.0],
        }
    )
    styled = style_technical_table(df.style)
    table = styled.to_html()
    assert _STYLE_UNFAVORABLE.split(";")[0] in table
    assert _STYLE_FAVORABLE.split(";")[0] in table

"""
Interactive Churn Prediction Dashboard — Dash (Plotly)
Run: python app/dashboard.py
Then open: http://127.0.0.1:8050
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = "data/processed/cleaned_telco.csv"

app = Dash(__name__)
app.title = "Churn Dashboard"


def load_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "20px"},
    children=[
        html.H1("Churn Client — Dashboard Interactif", style={"textAlign": "center"}),
        html.Hr(),

        # KPI row
        html.Div(id="kpi-row", style={"display": "flex", "gap": "20px", "marginBottom": "30px"}),

        # Controls
        html.Div([
            html.Label("Filtrer par type de contrat :"),
            dcc.Dropdown(
                id="contract-filter",
                options=[
                    {"label": "Tous", "value": "all"},
                    {"label": "Month-to-month", "value": "Month-to-month"},
                    {"label": "One year", "value": "One year"},
                    {"label": "Two year", "value": "Two year"},
                ],
                value="all",
                clearable=False,
                style={"width": "300px"},
            ),
        ], style={"marginBottom": "20px"}),

        # Charts row 1
        html.Div([
            dcc.Graph(id="churn-distribution", style={"flex": 1}),
            dcc.Graph(id="churn-by-contract", style={"flex": 1}),
        ], style={"display": "flex", "gap": "20px"}),

        # Charts row 2
        html.Div([
            dcc.Graph(id="monthly-charges-box", style={"flex": 1}),
            dcc.Graph(id="tenure-hist", style={"flex": 1}),
        ], style={"display": "flex", "gap": "20px", "marginTop": "20px"}),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("kpi-row", "children"),
    Output("churn-distribution", "figure"),
    Output("churn-by-contract", "figure"),
    Output("monthly-charges-box", "figure"),
    Output("tenure-hist", "figure"),
    Input("contract-filter", "value"),
)
def update_dashboard(contract_filter):
    df = load_df()

    if contract_filter != "all" and "Contract" in df.columns:
        df = df[df["Contract"] == contract_filter]

    churn_col = "Churn"
    churn_rate = df[churn_col].mean() * 100 if churn_col in df.columns else 0
    total = len(df)
    churned = int(df[churn_col].sum()) if churn_col in df.columns else 0

    # KPIs
    kpis = [
        _kpi_card("Total clients", f"{total:,}"),
        _kpi_card("Clients churned", f"{churned:,}"),
        _kpi_card("Taux de churn", f"{churn_rate:.1f}%"),
    ]

    # Churn distribution pie
    if churn_col in df.columns:
        counts = df[churn_col].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        counts["Churn"] = counts["Churn"].map({1: "Churned", 0: "Retained"})
        fig_pie = px.pie(counts, names="Churn", values="Count",
                         title="Répartition Churn / Non-Churn",
                         color_discrete_map={"Churned": "#EF553B", "Retained": "#636EFA"})
    else:
        fig_pie = go.Figure()

    # Churn by contract (only if column present)
    if "Contract" in df.columns and churn_col in df.columns:
        contract_churn = df.groupby("Contract")[churn_col].mean().reset_index()
        contract_churn.columns = ["Contract", "Churn Rate"]
        fig_contract = px.bar(contract_churn, x="Contract", y="Churn Rate",
                              title="Taux de Churn par Type de Contrat",
                              color="Churn Rate", color_continuous_scale="Reds")
    else:
        fig_contract = go.Figure()

    # Monthly charges boxplot
    if "MonthlyCharges" in df.columns and churn_col in df.columns:
        df_box = df.copy()
        df_box["Churn Label"] = df_box[churn_col].map({1: "Churned", 0: "Retained"})
        fig_box = px.box(df_box, x="Churn Label", y="MonthlyCharges",
                         title="Frais Mensuels — Churn vs Retained",
                         color="Churn Label",
                         color_discrete_map={"Churned": "#EF553B", "Retained": "#636EFA"})
    else:
        fig_box = go.Figure()

    # Tenure histogram
    if "tenure" in df.columns and churn_col in df.columns:
        df_hist = df.copy()
        df_hist["Churn Label"] = df_hist[churn_col].map({1: "Churned", 0: "Retained"})
        fig_tenure = px.histogram(df_hist, x="tenure", color="Churn Label",
                                  barmode="overlay", nbins=30,
                                  title="Distribution de l'Ancienneté (tenure)",
                                  color_discrete_map={"Churned": "#EF553B", "Retained": "#636EFA"})
    else:
        fig_tenure = go.Figure()

    return kpis, fig_pie, fig_contract, fig_box, fig_tenure


def _kpi_card(label: str, value: str):
    return html.Div(
        style={
            "background": "#f4f6f9",
            "borderRadius": "8px",
            "padding": "16px 24px",
            "minWidth": "150px",
            "textAlign": "center",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.1)",
        },
        children=[
            html.P(label, style={"margin": 0, "color": "#666", "fontSize": "13px"}),
            html.H3(value, style={"margin": "4px 0 0 0", "color": "#222"}),
        ],
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

"""
Dashboard interactif — Analyse du Churn Client
Lancement : python app/dashboard.py
URL : http://127.0.0.1:8050
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ── Palette & thème ───────────────────────────────────────────────────────────
ROUGE = "#ef4444"
BLEU = "#6366f1"
VERT = "#10b981"
ORANGE = "#f59e0b"
GRIS_FOND = "#f1f5f9"
BLANC = "#ffffff"
NAVY = "#0f172a"
TEXTE = "#1e293b"
TEXTE_SOFT = "#64748b"

COULEURS_CHURN = {"Résilié": ROUGE, "Fidèle": BLEU}

# Thème visuel partagé par tous les graphiques Plotly (dépaqueté via **TEMPLATE_CHART
# dans chaque update_layout) pour garantir un rendu homogène.
TEMPLATE_CHART = dict(
    paper_bgcolor=BLANC,
    plot_bgcolor=BLANC,
    font=dict(family="Inter, sans-serif", color=TEXTE, size=12),
    title_font=dict(size=14, color=TEXTE),
    title_x=0.01,
    title_xanchor="left",
    margin=dict(l=20, r=20, t=48, b=20),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=11),
    ),
    xaxis=dict(showgrid=False, zeroline=False, linecolor="#e2e8f0"),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False),
)

EXTERNAL = ["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"]

# ── App ───────────────────────────────────────────────────────────────────────
# Le dashboard lit le CSV brut (et non les données déjà encodées du modèle) car il
# a besoin des libellés lisibles pour l'affichage et le filtre par contrat.
DATA_PATH = "data/raw/telco_customer_churn.csv"

app = Dash(__name__, external_stylesheets=EXTERNAL)
app.title = "Churn Analytics"


def load_df() -> pd.DataFrame:
    """Charge et nettoie a minima le dataset pour l'affichage."""
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


# ── Composants réutilisables ──────────────────────────────────────────────────
def kpi_card(label, valeur, sous_titre="", couleur_accent=BLEU):
    """Carte d'indicateur clé (KPI) : libellé, valeur, sous-titre, accent coloré."""
    return html.Div(
        style={
            "background": BLANC,
            "borderRadius": "14px",
            "padding": "20px 24px",
            "flex": 1,
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04)",
            "borderLeft": f"4px solid {couleur_accent}",
            "minWidth": "160px",
        },
        children=[
            html.P(
                label,
                style={
                    "margin": 0,
                    "color": TEXTE_SOFT,
                    "fontSize": "12px",
                    "fontWeight": "500",
                    "letterSpacing": "0.05em",
                    "textTransform": "uppercase",
                },
            ),
            html.H2(
                valeur,
                style={
                    "margin": "8px 0 4px",
                    "color": TEXTE,
                    "fontSize": "28px",
                    "fontWeight": "700",
                },
            ),
            html.P(sous_titre, style={"margin": 0, "color": TEXTE_SOFT, "fontSize": "12px"}),
        ],
    )


def section_titre(texte):
    """Titre de section (H3) au style homogène."""
    return html.H3(
        texte,
        style={
            "color": TEXTE,
            "fontSize": "15px",
            "fontWeight": "600",
            "marginBottom": "12px",
            "marginTop": "0",
            "letterSpacing": "-0.01em",
        },
    )


def carte_graphique(graph_id):
    """Enveloppe visuelle standard autour d'un graphique Plotly."""
    return html.Div(
        style={
            "background": BLANC,
            "borderRadius": "14px",
            "padding": "16px",
            "flex": 1,
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04)",
        },
        children=[dcc.Graph(id=graph_id, config={"displayModeBar": False})],
    )


# ── Layout ────────────────────────────────────────────────────────────────────
# Les zones dynamiques (KPIs, figures) sont déclarées vides ici et remplies par le
# callback update_dashboard() à chaque changement du filtre.
app.layout = html.Div(
    style={"fontFamily": "Inter, sans-serif", "backgroundColor": GRIS_FOND, "minHeight": "100vh"},
    children=[
        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {NAVY} 0%, #1e3a5f 100%)",
                "padding": "24px 40px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "boxShadow": "0 4px 24px rgba(0,0,0,0.15)",
            },
            children=[
                html.Div(
                    [
                        html.H1(
                            "Churn Analytics",
                            style={
                                "margin": 0,
                                "color": BLANC,
                                "fontSize": "22px",
                                "fontWeight": "700",
                                "letterSpacing": "-0.02em",
                            },
                        ),
                        html.P(
                            "Analyse du comportement de résiliation — Telco Customer Dataset",
                            style={
                                "margin": "4px 0 0",
                                "color": "rgba(255,255,255,0.55)",
                                "fontSize": "13px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "12px"},
                    children=[
                        html.Div(
                            "Filtrer par contrat",
                            style={
                                "color": "rgba(255,255,255,0.7)",
                                "fontSize": "13px",
                                "fontWeight": "500",
                            },
                        ),
                        dcc.Dropdown(
                            id="contract-filter",
                            options=[
                                {"label": "Tous les contrats", "value": "all"},
                                {"label": "Month-to-month", "value": "Month-to-month"},
                                {"label": "One year", "value": "One year"},
                                {"label": "Two year", "value": "Two year"},
                            ],
                            value="all",
                            clearable=False,
                            style={"width": "220px", "fontSize": "13px"},
                        ),
                    ],
                ),
            ],
        ),
        # ── Corps ───────────────────────────────────────────────────────────
        html.Div(
            style={"padding": "28px 40px", "maxWidth": "1400px", "margin": "0 auto"},
            children=[
                # KPIs — contenu généré dynamiquement par le callback
                html.Div(
                    id="kpi-row", style={"display": "flex", "gap": "16px", "marginBottom": "28px"}
                ),
                # Ligne 1 — Pie + Contrat
                html.Div(
                    style={"marginBottom": "8px"},
                    children=[
                        section_titre("Vue d'ensemble"),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                    children=[carte_graphique("churn-pie"), carte_graphique("churn-by-contract")],
                ),
                # Ligne 2 — Charges + Tenure
                html.Div(
                    style={"marginBottom": "8px"},
                    children=[
                        section_titre("Comportement financier & ancienneté"),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                    children=[
                        carte_graphique("monthly-charges-box"),
                        carte_graphique("tenure-hist"),
                    ],
                ),
                # Ligne 3 — Internet + Senior
                html.Div(
                    style={"marginBottom": "8px"},
                    children=[
                        section_titre("Profil client"),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "gap": "16px", "marginBottom": "32px"},
                    children=[
                        carte_graphique("churn-by-internet"),
                        carte_graphique("churn-by-senior"),
                    ],
                ),
                # Footer — chaîne concaténée sur deux lignes pour respecter la
                # limite de longueur de ligne, sans changer le texte affiché.
                html.Div(
                    "Projet #3 — Introduction au Machine Learning · L'École Multimédia"
                    " · Kouame Adamou Kevin",
                    style={
                        "textAlign": "center",
                        "color": TEXTE_SOFT,
                        "fontSize": "12px",
                        "paddingTop": "16px",
                        "borderTop": "1px solid #e2e8f0",
                    },
                ),
            ],
        ),
    ],
)


# ── Callback principal ────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-row", "children"),
    Output("churn-pie", "figure"),
    Output("churn-by-contract", "figure"),
    Output("monthly-charges-box", "figure"),
    Output("tenure-hist", "figure"),
    Output("churn-by-internet", "figure"),
    Output("churn-by-senior", "figure"),
    Input("contract-filter", "value"),
)
def update_dashboard(contract_filter):
    """Recalcule les KPIs et régénère les 6 figures selon le contrat sélectionné."""
    df = load_df()
    if contract_filter != "all":
        df = df[df["Contract"] == contract_filter]

    total = len(df)
    churned = int(df["Churn"].sum())
    taux = df["Churn"].mean() * 100
    charges_moy = df["MonthlyCharges"].mean()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    kpis = [
        kpi_card("Total clients", f"{total:,}", "dans la sélection", BLEU),
        kpi_card("Clients résiliés", f"{churned:,}", f"sur {total:,}", ROUGE),
        kpi_card("Taux de churn", f"{taux:.1f} %", "taux de résiliation", ORANGE),
        kpi_card("Charge mensuelle moy.", f"{charges_moy:.1f} €", "par client actif", VERT),
    ]

    # ── Pie donut ─────────────────────────────────────────────────────────────
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Nombre"]
    counts["Churn"] = counts["Churn"].map({1: "Résilié", 0: "Fidèle"})
    fig_pie = px.pie(
        counts,
        names="Churn",
        values="Nombre",
        title="Répartition Churn / Non-Churn",
        color="Churn",
        color_discrete_map=COULEURS_CHURN,
        hole=0.55,
    )
    fig_pie.update_traces(
        textposition="outside", textfont_size=12, marker=dict(line=dict(color=BLANC, width=2))
    )
    fig_pie.update_layout(**TEMPLATE_CHART)

    # ── Bar contrat ────────────────────────────────────────────────────────────
    contract_churn = df.groupby("Contract")["Churn"].mean().reset_index()
    contract_churn.columns = ["Contrat", "Taux"]
    contract_churn["Taux"] = (contract_churn["Taux"] * 100).round(1)
    contract_churn = contract_churn.sort_values("Taux", ascending=True)
    # go.Bar (API bas niveau) plutôt que px.bar : nécessaire pour combiner une
    # échelle de couleur continue avec du texte affiché sur chaque barre.
    fig_contract = go.Figure(
        go.Bar(
            x=contract_churn["Taux"],
            y=contract_churn["Contrat"],
            orientation="h",
            marker=dict(
                color=contract_churn["Taux"],
                colorscale=[[0, "#dbeafe"], [0.5, "#6366f1"], [1, ROUGE]],
                line=dict(width=0),
            ),
            text=[f"{v:.1f} %" for v in contract_churn["Taux"]],
            textposition="outside",
            textfont=dict(size=12, color=TEXTE),
        )
    )
    fig_contract.update_layout(
        title="Taux de Churn par Type de Contrat",
        xaxis_title="Taux de churn (%)",
        **TEMPLATE_CHART,
    )

    # ── Boxplot frais ─────────────────────────────────────────────────────────
    df_box = df.copy()
    df_box["Statut"] = df_box["Churn"].map({1: "Résilié", 0: "Fidèle"})
    fig_box = px.box(
        df_box,
        x="Statut",
        y="MonthlyCharges",
        title="Frais Mensuels — Résilié vs Fidèle",
        color="Statut",
        color_discrete_map=COULEURS_CHURN,
        points="outliers",
    )
    fig_box.update_traces(marker=dict(size=3, opacity=0.5), line=dict(width=1.5))
    fig_box.update_layout(
        yaxis_title="Frais mensuels (€)",
        showlegend=False,
        **TEMPLATE_CHART,
    )

    # ── Histogram tenure ──────────────────────────────────────────────────────
    df_hist = df.copy()
    df_hist["Statut"] = df_hist["Churn"].map({1: "Résilié", 0: "Fidèle"})
    fig_tenure = px.histogram(
        df_hist,
        x="tenure",
        color="Statut",
        barmode="overlay",
        nbins=30,
        title="Distribution de l'Ancienneté",
        color_discrete_map=COULEURS_CHURN,
        opacity=0.75,
    )
    fig_tenure.update_layout(
        xaxis_title="Ancienneté (mois)",
        yaxis_title="Nombre de clients",
        bargap=0.02,
        **TEMPLATE_CHART,
    )

    # ── Bar service internet ───────────────────────────────────────────────────
    internet_churn = df.groupby("InternetService")["Churn"].mean().reset_index()
    internet_churn.columns = ["Service", "Taux"]
    internet_churn["Taux"] = (internet_churn["Taux"] * 100).round(1)
    internet_churn = internet_churn.sort_values("Taux", ascending=False)
    fig_internet = px.bar(
        internet_churn,
        x="Service",
        y="Taux",
        title="Taux de Churn par Service Internet",
        color="Taux",
        color_continuous_scale=[[0, "#dbeafe"], [0.5, "#6366f1"], [1, ROUGE]],
        text=[f"{v:.1f} %" for v in internet_churn["Taux"]],
    )
    fig_internet.update_traces(textposition="outside", textfont=dict(size=12), marker_line_width=0)
    fig_internet.update_layout(
        yaxis_title="Taux de churn (%)",
        coloraxis_showscale=False,
        **TEMPLATE_CHART,
    )

    # ── Bar Senior vs Non-Senior ──────────────────────────────────────────────
    df_s = df.copy()
    df_s["Profil"] = df_s["SeniorCitizen"].map({1: "Senior", 0: "Non-Senior"})
    senior_churn = df_s.groupby("Profil")["Churn"].mean().reset_index()
    senior_churn.columns = ["Profil", "Taux"]
    senior_churn["Taux"] = (senior_churn["Taux"] * 100).round(1)
    fig_senior = px.bar(
        senior_churn,
        x="Profil",
        y="Taux",
        title="Taux de Churn — Senior vs Non-Senior",
        color="Profil",
        color_discrete_map={"Senior": ROUGE, "Non-Senior": BLEU},
        text=[f"{v:.1f} %" for v in senior_churn["Taux"]],
    )
    fig_senior.update_traces(textposition="outside", textfont=dict(size=12), marker_line_width=0)
    fig_senior.update_layout(
        yaxis_title="Taux de churn (%)",
        showlegend=False,
        **TEMPLATE_CHART,
    )

    return kpis, fig_pie, fig_contract, fig_box, fig_tenure, fig_internet, fig_senior


# ── Lancement ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


app = dash.Dash(__name__)
server = app.server

# Load the dataset
train_data = pd.read_csv("train.csv")
train_data.drop('Id', axis=1, inplace=True)

# Feature Engineering
def feature_engineering(df):
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df.get("2ndFlrSF", 0)
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['Remodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
    df['AvgQual_Neigh'] = df.groupby('Neighborhood')['OverallQual'].transform('mean')
    return df

train_data = feature_engineering(train_data)

# Selected features
selected_features = [
    'OverallQual', 'YearBuilt', 'YearRemodAdd', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',
    'LotFrontage', 'LotArea', 'MSSubClass', 'OverallCond', 'MSZoning', 'LotShape', 'LandContour',
    'LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st',
    'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType',
    'SaleCondition', 'TotalSF', 'TotalBath', 'TotalPorchSF', 'HouseAge', 'Remodeled', 'AvgQual_Neigh'
]

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Feature Selection Analysis"

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Feature Selection Analysis"), className="mb-4")),
    
    dbc.Row(dbc.Col(html.H3("Introduction"), className="mb-3")),
    dbc.Row(dbc.Col(html.P("""
        Cette application présente le processus de sélection des caractéristiques pour la prédiction des prix de maisons.
        Nous avons analysé la corrélation des variables numériques, vérifié la multicolinéarité, et exploré les distributions
        des variables catégoriques pour sélectionner les plus pertinentes.
    """))),
    
    dbc.Tabs([
        dbc.Tab(label="Corrélation et Scatter Plot", children=[
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='scatter-feature-dropdown', 
                                     options=[{'label': col, 'value': col} for col in train_data[selected_features].select_dtypes('number')],
                                     value='GrLivArea', clearable=False), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='scatter-plot'), width=6),
                dbc.Col(dcc.Graph(id='corr-bar-chart'), width=6)
            ]),
        ]),
        
        dbc.Tab(label="Sélection des Variables Catégoriques", children=[
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='cat-feature-dropdown', 
                                     options=[{'label': col, 'value': col} for col in selected_features if train_data[col].dtype == 'object'],
                                     value='Neighborhood', clearable=False), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='cat-boxplot'), width=6),
                dbc.Col(dcc.Graph(id='cat-piechart'), width=6)
            ])
        ])
    ])
])

# Callbacks
@app.callback(
    [Output('scatter-plot', 'figure'), Output('corr-bar-chart', 'figure')],
    Input('scatter-feature-dropdown', 'value')
)
def update_scatter_plot(selected_feature):
    scatter_fig = px.scatter(train_data, x=selected_feature, y='SalePrice', trendline='ols')
    scatter_fig.update_layout(title=f"Scatter Plot of {selected_feature} vs SalePrice")
    
    corr_value = train_data[[selected_feature, 'SalePrice']].select_dtypes('number').corr().iloc[0, 1]
    corr_fig = px.bar(x=[selected_feature], y=[corr_value], labels={'x': 'Feature', 'y': 'Correlation'})
    corr_fig.update_layout(title=f"Correlation of {selected_feature} with SalePrice")
    
    return scatter_fig, corr_fig

@app.callback(
    [Output('cat-boxplot', 'figure'), Output('cat-piechart', 'figure')],
    Input('cat-feature-dropdown', 'value')
)
def update_cat_plots(selected_feature):
    box_fig = px.box(train_data, x=selected_feature, y='SalePrice')
    box_fig.update_layout(title=f"Box Plot of {selected_feature} vs SalePrice")
    
    value_counts = train_data[selected_feature].value_counts(normalize=True)
    pie_fig = px.pie(names=value_counts.index, values=value_counts.values, 
                      title=f"Proportion of Unique Values in {selected_feature}")
    
    return box_fig, pie_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

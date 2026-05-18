import warnings
warnings.filterwarnings("ignore")

from dash import Dash, dcc, html, Input, Output, callback, no_update
# from IPython.display import Markdown, display
from IPython.display import display_markdown
import time

import plotly.graph_objects as go

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sklearn.metrics

import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

import matplotlib as plt
import seaborn as sns
import os

#--- data gen as input file
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/AirPassengers.csv")
print([df.columns])

### predpare 
df_rowsnull = pd.DataFrame(columns=df.columns)

for i in range(len(df)):
    df_rowsnull.loc[i] = [0, None] ###[0, 0]
        
df_rowsnull

df = pd.concat([df, df_rowsnull], axis=0, ignore_index=True)
period_index= pd.period_range(start='1949-01', periods=len(df), freq='M')
df['date'] = period_index.to_timestamp()
df.set_index('date', inplace=True)
df.sort_values(by=['date'], ascending=True, inplace=True)
df_train= df.iloc[0:144]

order= np.array(range(0, 13)) 

# ## Functions to estimate and predict using sARIMAx modeling
# check_correlation(df, col='value')
def check_seasonality(df, col='value'):
    decomposition= seasonal_decompose(df[col],
                                       model='additive',
                                       period=12)
    decomposition.plot()
    return decomposition
# check_seasonality(df_train, col='value')

def adfuller_test(df, col='value'):
    """
    ADFuller test, also called "Augmented Dickey-Fuller test
    Tests that Null hypothesis if or not a unit root presents in TS data
    Null hypothesis: data is non-stationary if a unit root presents
    Alternative HYPO. with less than p-value, meaning the stationary data
    """
    white_test=adfuller(df[col], autolag='AIC')
    return (print(f"ADFuller stat(AIC): {white_test[0]:.2f}"),
    print(f"P-value: {white_test[1]:.3f}"),
    print(f"No. of Lag: {white_test[2]}"),
    print(f"No. of Obs.: {white_test[3]}"),
    print(f"Critical P-value:"),
    )

def adfuller_test_diff(df, col='value', period=1):
    """difference to remove non-stationary so called noise of data series"""
    df_diff=df[col].diff(periods=period).fillna(df[col]).reset_index()
    df_diff=df_diff.set_index('date')
    print(f"Adfuller tested with diffrences between 0th and {str(period)}nd/th Obs. \n")
    adfuller_test(df_diff, col=col) # autolag='AIC'
#
def model_summary(df, col='value', predict_n= len(df_train), order=(6,1,3), seasonal_order=(1,0,1, 12)):
    sarima= SARIMAX(df[col], order=order, seasonal_order=seasonal_order, trend='ct')
    model=sarima.fit()
    return model.summary()

def plot_predicted(df, df_train, col='value', 
                   predict_n=len(df_train),
                   order=(6,1,3), 
                   seasonal_order=(1,0,1,12),
                   ):
    """
    Plot predicted value in future and past together using Plotly library
    """
    import plotly.graph_objects as go
    predict_n2= len(df) # - (predict_n -1)
    sarima= SARIMAX(df[col], 
                    order=order,
                    seasonal_order=seasonal_order,
                    trend='ct'
                    )
    model= sarima.fit()
    
    df_forecasted = []
    df_forecasted=model.predict( len(df_train)- 1).round(2) #sarima.fit().predict()
    df_forecasted.to_csv('../data/df_forecasted1.csv')
    df_forecasted2 = pd.DataFrame(df_forecasted, columns=['date', 'predicted_mean'])
    #-
    x= df_forecasted2.index
    y= df_forecasted2['predicted_mean'] #[predict]
    ci_upper= y + 1.96 * np.std(y) / np.sqrt(len(y))
    ci_lower= y - 1.96 * np.std(y) / np.sqrt(len(y))
    ###-
    fig= go.Figure()
    fig.add_trace(go.Scatter(
        x=df_train.index,
        y=df_train[col],
        mode='lines+markers',
        line=dict(width=1, color='blue'),
        marker=dict(size=1, color='orange',
                    line=dict(width=0.3, color='gray')),
        name='Raw Data',
    ))
    #-
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([ci_upper, ci_lower[::-1]]),
        fill='toself',
        fillcolor='darkgreen', #'rgb(0, 100, 80, 0.9)',
        line=dict(color='rgb(255, 255, 255, 0)', width=0.01), ### 0.1),
        hoverinfo='skip',
        showlegend=True,
        name='95% CI',
    ))
    #-
    fig.add_trace(go.Scatter(
        x=df_forecasted2.index,
        y=df_forecasted2['predicted_mean'], ###df_forecasted[col],
        mode='lines+markers',
        line=dict(width=1.3, color='orangered'),
        marker=dict(size=1.2, color='orange',
                      line=dict(width=0.3, color='gray')),
        name='Predicted Data',
    ))
    fig.update_layout(
        width= 900, #600, 
        height=600, #450,
        title="<b>Prediction of Air Passengers.csv: Time-series using sARIMAx<br><sup>order="+str(order)+", seasonal order"+str(seasonal_order) +"</sup></b>",
        xaxis_tickformat = '%d %B<br>%Y', # xaxis_tickformat = '%d %B (%a)<br>%Y'
        xaxis_title='Date',
        yaxis_title='No. of Air Passengers', #'Monthly Amount',
    )
    return fig

def cross_validation(df=df_train, order=(2,1,3), seasonal_order=(2,1,3,12)):
    #
    mae_lst, mse_lst, rmse_lst, mape_lst = [], [], [], []
    # Parameters for rolling window
    window_size = 60 #30 ### 60  # Size of the initial training window
    test_size = 24 #6 ### 12    # Size of the test set for each rolling step\

    X = df_train.values
    # Loop through the data with a rolling window approach
    for start in range(0, len(X) - window_size - test_size + 1):
        # split both train and test datasets
        train = X[start:start + window_size]
        test = X[start + window_size:start + window_size + test_size]
        # SARIMAX model on the train dataset
        model = SARIMAX(train, order= order, #(2, 1, 3), 
                        seasonal_order= seasonal_order, #(2, 1, 3, 12), 
                        trend='ct', enforce_invertibility= False )
        model_res = model.fit(maxiter=300, method='nm') # model.fit()
        # prediction of test dataset
        prediction = model_res.predict(start=len(train), end=len(train) + len(test) - 1)
        # indicators to evaluate raw and prediction
        mae = mean_absolute_error(test, prediction)
        mse = mean_squared_error(test, prediction)
        rmse = np.sqrt(mse)
        mape = np.nanmean(np.abs((test - prediction) / test)) * 100
        
        # appending each metrics
        mae_lst.append(mae)
        mse_lst.append(mse)
        rmse_lst.append(rmse)
        mape_lst.append(mape)
        #
    
    # Calculate and print average values across all splits
    data_metrics = {
        "order": [str(order)],
        "seasonal_order": [str(seasonal_order)],
        "MSE": [np.nanmean(mse_lst)],
        "RMSE": [np.nanmean(rmse_lst)],
        "MAE": [np.nanmean(mae_lst)],
        "MAPE": [np.nanmean(mape_lst)]}
    df_metrics = pd.DataFrame(data_metrics).round(2)
    metrics_mrk= df_metrics.to_markdown() #(index=False, tablefmt='grid')
    
    return df_metrics, metrics_mrk, model_res # df_metrics

# autocorrelation: acf, pacf
from statsmodels.tsa.stattools import acf, pacf

def create_corr_plot_diff(df_train, col='value', period=12, plot_pacf=False, diff='(1)'):
    if period==None:
        series= df_train
    else:
        df_diff=df_train[col].diff(periods=period).fillna(df_train[col]).reset_index()
        series=df_diff.set_index('date')
    #---
    corr_array = pacf(series.dropna(), alpha=0.05) if plot_pacf else acf(series.dropna(), alpha=0.05)
    # print(corr_array)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), mode='lines',line_color='#3f3f3f') 
    for x in range(len(corr_array[0]))]
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=corr_array[0], mode='markers', marker_color='#1f77b4', marker_size=6)
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=upper_y, mode='lines', line_color='rgba(255,255,255,0)')
    fig.add_scatter(x=np.arange(len(corr_array[0])), y=lower_y, mode='lines',fillcolor='rgba(32, 146, 230,0.3)',
            fill='tonexty', line_color='rgba(255,255,255,0)')
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1,42])
    fig.update_yaxes(zerolinecolor='#000000')
    
    title='Partial Autocorrelation (PACF): '+str(diff) if plot_pacf else 'Autocorrelation (ACF): '+str(diff)
    fig.update_layout(title=title,
                    width=900, #600, 
                    height=240, #300
                    )
    # fig.show()
    return fig

#--- plot Graph
app = Dash(__name__)
app.layout= html.Div([
    html.Div([
        html.Br(),
        dcc.Dropdown(
            [ 'order(2,1,3)-start', #f"order{optimal_order2}-Optimal",
            'order(0,0,0)', 'order(1,0,1)', #'order(2,1,3)',
            'order(3,0,3)', 'order(3,1,3)',
             'order(4,0,3)', 'oder(5,0,3)','order(6,0,3)', 'order(6,1,3)', 'order(6,1,6)',
             'order(7,1,3)', 'order(8,0,3)', 'order(9,0,3)', 'order(9,1,3)',
             'order(9,2,3)', 'order(9,3,3)'],
            'order(2,1,3)-start', #f"order{optimal_order2}-Optimal", #?'order(2,1,3)', #'order(3,1,3)', ###'order(6,1,3)',
            id='order'
        ),
        ], style={
            'Color': '#212121',
            'backgroundColor': '#212121',
            'width': '100%', #'40vH', 
            'height': '40px'
                  }
             ),
    
    html.Div([
        html.Br(),
        dcc.Dropdown(
            [ 'seasonal_order(2,1,3,12)-start', #f"seasonal_order{optimal_seasonality2}-Optimal" ,
             'seasonal_order(1,0,1,12)', 'seasonal_order(2,1,1,12)',#'seasonal_order(2,1,3,12)', 
             'seasonal_order(3,1,3,12)','seasonal_order(6,1,3,12)', 'seasonal_order(6,1,6,12)'],
             'seasonal_order(2,1,3,12)-start', #f"seasonal_order{optimal_seasonality2}-Optimal", #'seasonal_order(2,1,1,12)',
            id='seasonal_order'
        ),
    ], style={'width': '100%', #48%', 
              'display': 'inline-block'}), # 'float': 'right'})
    
    # Tabs
    # html.Div([
        html.Br(),
        dcc.Tabs(id="tabs", value='tab-1', 
                 children=[dcc.Tab(label='Tab 1: Prediction', value='tab-1'),
                           dcc.Tab(label='Tab 2: Results', value='tab-2'),
                           dcc.Tab(label='Tab-3: Plain Autocorrelation', value='tab-3'),
                           dcc.Tab(label='Tab-4: Lagging Autocorrelation', value='tab-4')
        ],  style={'width': '100%', 'display': 'inline-block'}),
    html.Div(id='content')
])

@app.callback(
    Output('content', 'children'),
    [Input("order", 'value'),
    Input('seasonal_order', 'value'),
    Input('tabs', 'value'),
    ])
    
#
def update_graph( order_value, seasonal_order_value, tab):
    np.set_printoptions(suppress=True)
    order_value2 = order_value[6:len(order_value)]
    seasonal_order_value2 = seasonal_order_value[15:len(seasonal_order_value)-1]
    print(( int(order_value2[0]), int(order_value2[2]), int(order_value2[4])))
    print(( int(seasonal_order_value2[0]), int(seasonal_order_value2[2]), int(seasonal_order_value2[4]), 12))

    # tab
    if tab == 'tab-1':
        print("tab-1")
        fig=plot_predicted(df, df_train, col='value', 
            predict_n=len(df_train),
            order= ( int(order_value2[0]), int(order_value2[2]), int(order_value2[4])), #(4,0,3), 
            seasonal_order= (int(seasonal_order_value2[0]), int(seasonal_order_value2[2]), int(seasonal_order_value2[4]), 12))
        #
        path_save = "./outputs"
        file_html = "./Mycodes_sARIMx_v1_20250825.html"
        
        if not os.path.exists(path_save):
            os.mkdir(path_save)
            fig.write_html(path_save+file_html, full_html=False, include_plotlyjs='cdn')
        else:
            fig.write_html(path_save+file_html, full_html=False, include_plotlyjs='cdn')
        
        return dcc.Graph(figure=fig)
    elif tab == 'tab-2':
        print('tab-2')
        #--- adfuller_test()
        white_test=adfuller(df_train['value'], autolag='AIC')
        # adfuller test with differenced data
        df_diff=df_train['value'].diff(periods=1).fillna(df_train['value']).reset_index()
        df_diff=df_diff.set_index('date')
        white_diff1_test=adfuller(df_diff['value'], autolag='AIC')
        #---sARIMAx Model summary
        model_res = model_summary(df_train, col='value', predict_n= len(df_train), 
                            order= ( int(order_value2[0]), int(order_value2[2]), int(order_value2[4])),
                            seasonal_order= (int(seasonal_order_value2[0]), int(seasonal_order_value2[2]), int(seasonal_order_value2[4]), 12))
        #--- Metrics using Cross Evaluation
        df_metrics, metrics_mrk, model_res = cross_validation(df=df_train, 
                                                order= ( int(order_value2[0]), int(order_value2[2]), int(order_value2[4])), #(4,0,3), 
                                                seasonal_order= (int(seasonal_order_value2[0]), int(seasonal_order_value2[2]), int(seasonal_order_value2[4]), 12))
        #
        lines_md = [
            f"Results (sarimax model):",
            f"\n",
            f"```\n" + model_res.summary().as_text() + "\n```",
            f"\n",
            f"**White Noise Test: Difference(1) with lag 1 recommend in this case**",
            f"\n"
            f"**difference(0)**: ADFuller stat(AIC): {white_test[0]:.2f}, **P-value**: {white_test[1]:5f}, No. of Lag: {white_test[2]}, No. of Obs.: {white_test[3]}", 
            f"\n",
            f"**difference(1)**: ADFuller stat(AIC): {white_diff1_test[0]:.2f}, **P-value**: {white_diff1_test[1]:.3f}, No. of Lag: {white_diff1_test[2]}, No. of Obs.: {white_diff1_test[3]}",
            f" \n ",
            f"** Metrics for cross evaluation**:",
            f"\n",
            f"{df_metrics.to_markdown(index=False)}"
        ]
        return dcc.Markdown("\n".join(lines_md))
    #
    elif tab == 'tab-3':
        print('Tab-3')
        fig_01 = create_corr_plot_diff(df_train, col='value', period=None, plot_pacf=False, diff='(None lagged or difference)')
        fig_02 = create_corr_plot_diff(df_train, col='value', period=None, plot_pacf=True, diff='(None lagged or difference)')
        return dcc.Graph(figure=fig_01), dcc.Graph(figure=fig_02) #, dcc.Graph(figure=fig_corr1) #decomposition.plot())
        #
    elif tab == 'tab-4':
        print('Tab-4')
        fig_11 = create_corr_plot_diff(df_train, col='value', period=12, plot_pacf=False, diff='(1 lagged or difference)')
        fig_12 = create_corr_plot_diff(df_train, col='value', period=12, plot_pacf=True, diff='(1 lagged or difference)')
        return dcc.Graph(figure=fig_11), dcc.Graph(figure=fig_12)
    #
    return html.Div('invalid Tab!')
#---
if __name__ == ('__main__'):
    app.run(debug=True)
import pandas as pd
from textblob import TextBlob
import preprocessor as p
from fbprophet import Prophet
import plotly.graph_objects as go
import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
from fbprophet import Prophet
import warnings
from crypto_news_api import CryptoControlAPI
css = ["https://stackpath.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css"]
app = dash.Dash(__name__, external_stylesheets=css)
warnings.simplefilter(action='ignore')

api_key_news = '2df3a15efa6332e41ba90795b9639d32'
api = CryptoControlAPI(api_key_news)


latestTweets = api.getLatestTweetsByCoin("bitcoin")
tweets = pd.DataFrame(latestTweets)
tweets = tweets[['publishedAt','text']]
tweets['publishedAt'] = pd.to_datetime(tweets['publishedAt'])
list1 = []
for i in tweets['publishedAt']:
    i = i.strftime('%d-%m-%Y')
    list1.append(i)
tweets['publishedAt'] = list1
tweets['publishedAt'] = pd.to_datetime(tweets['publishedAt'])
tweets.rename(columns = {'publishedAt' : 'Date'}, inplace=True)
tweets['cleaned'] = tweets.text.apply(p.clean)


test_list = []
for i in tweets.cleaned:
    i = TextBlob(i).sentiment.polarity
    test_list.append(i)
tweets['polarity'] = test_list


sentiment = []
for i in tweets.polarity:
    if i>0:
        sentiment.append('positive')
    elif i == 0 :
        sentiment.append('neutral')
    elif i<0 :
        sentiment.append('negative')
tweets['sentiment'] = sentiment
tweet_chart = tweets.groupby('Date',as_index=False)['polarity'].mean()
tweet_table = tweets[['Date','text','sentiment']]
tweet_table.rename(columns={'text' : 'Tweets'},inplace=True)


df = pd.read_csv('http://data.bitcoinity.org/export_data.csv?currency=USD&data_type=price&exchange=coinbase&r=hour&t=l&timespan=30d',
                 parse_dates=['Time'])
df.set_index('Time', inplace=True)
df.head()

df['ds'] = df.index
df['y'] = df['avg']
forecast_data = df[['ds', 'y']].copy()
forecast_data.reset_index(inplace=True)
del forecast_data['Time']

df_list = []
for i in forecast_data.ds:
    i = i.strftime('%Y-%m-%d')
    df_list.append(i)
forecast_data.ds = df_list
forecast_data.ds = pd.to_datetime(forecast_data.ds)
forecast_data = forecast_data.groupby('ds',as_index=False)['y'].mean()

m = Prophet(weekly_seasonality=True)
m.fit(forecast_data)
future = m.make_future_dataframe(periods=96, freq='H')
future.tail()

forecast = m.predict(future)

#charts :
# pie charts:
pos = 0
neg = 0
neu = 0
for i in tweets.sentiment:
    if i == 'positive':
        pos=pos+1
    elif i == 'negative':
        neg=neg+1
    elif i == 'neutral':
        neu+=1
groups = tweets.sentiment.unique().tolist()
values = [pos,neg,neu]
pie_chart  = go.Figure(data=[go.Pie(labels=groups, values=values
                            )])

# line chart for hourly btc price :
df_chart = df
df_chart.ds = pd.to_datetime(df_chart.ds, unit="s")
btc_chart = go.Figure({
        'data': [
            {
                'type': 'scatter',
                'x': df_chart.ds,
                'y': df_chart.y,
                'mode': 'lines',
                'marker': {'color': '#036bfc'},
                'hoverlabel': {'namelength': 25}
            }
        ],
        'layout': {
            'autosize': True,
            'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
            'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
            'yaxis': {'title': {'text': 'Price in USD'},
                      'gridcolor': '#f5f5f5'
                      },
            'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'hovermode': 'x',
            'dragmode': False
        }
    })


# line chart for twitter sentiment :
sent_chart = go.Figure({
        'data': [
            {
                'type': 'scatter',
                'x': tweet_chart.Date,
                'y': tweet_chart.polarity,
                'mode': 'lines+markers',
                'marker': {'color': '#036bfc'},
                'hoverlabel': {'namelength': 25}
            }
        ],
        'layout': {
            'autosize': True,
            'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
            'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
            'yaxis': {'title': {'text': 'Polarity'},
                      'gridcolor': '#f5f5f5'
                      },
            'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'hovermode': 'x',
            'dragmode': False
        }
    })

# predictions charts :


predict_chart = go.Figure({
        'data': [
            {
                'type': 'scatter',
                'x': forecast.ds[len(forecast_data)-1:],
                'y': forecast.yhat[len(forecast_data)-1:],
                'mode': 'lines',
                'name' : 'Prediction',
                #'marker': {'color': '#036bfc'},
                'hoverlabel': {'namelength': 25}
            },
            {
                'type': 'scatter',
                'x': forecast_data.ds,
                'y': forecast_data.y,
                'mode': 'lines',
                'name' : 'Actual data',
                #'marker': {'color': '#036bfc'},
                'hoverlabel': {'namelength': 25}
            }
        ],
        'layout': {
            'autosize': True,
            'legend': {'bgcolor': 'rgba(255,255,255,0)', 'x': 0, 'y': 1},
            'xaxis': {'tickformat': '%m-%d', 'title': {'text': "Date"}},
            'yaxis': {'title': {'text': 'Price in USD'},
                      'gridcolor': '#f5f5f5'
                      },
            'margin': {"r": 0, "t": 10, "l": 60, "b": 50},
            'plot_bgcolor': 'rgba(255,255,255,1)',
            'paper_bgcolor': 'rgba(255,255,255,1)',
            'hovermode': 'x',
            'dragmode': False
        }
    })

app.layout = html.Div([
    html.Div(
        html.H3('How does social media affect Bitcoin price movement?'),
        style={"text-align": "center"}
    ),
    html.Br(),
    html.Div(
        [
            html.Div([
                html.H6('Bitcoin Prices latest 30 days',
                        style={'text-align': 'center'}
                        ),
                dcc.Graph(
                    figure=btc_chart
                )
            ],className='grid-item')
        ],className="grid-container-one-col"),
    html.Br(),
    html.Div([
        html.Div([
            html.H6('Sentiment Chart on most recent tweets about bitcoin',
                    style={'text-align': 'center'}),
            dcc.Graph(
                figure=sent_chart
            )
        ],className='grid-item'),
        html.Div([
            html.H6('Percentage of each sentiment in the tweets',
                    style={'text-align': 'center'}),
            dcc.Graph(
                figure=pie_chart
            )
        ],className='grid-item')
    ],className='grid-container-two-cols'),
    html.Div([
        html.Div([
            html.H6('Predictions on future bitcoin Price',
                    style={'text-align': 'center'}),
            dcc.Graph(
                figure=predict_chart
            )
        ],className='grid-item')
    ],className='grid-container-one-col'),
    html.Br(),
    html.Div([
        dt.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in tweet_table.columns],
            data=tweet_table.to_dict('records'),
            style_data={
                'whiteSpace': 'normal',
                'lineHeight': '15px',
                'height': 'auto', 'overflowY': 'auto'
            },
            style_cell={'textAlign': 'left'},
            page_action='native',
            page_size=10,
            page_current=0
        )
    ],className='grid-item')
],className='container')

if __name__ ==  '__main__':
    app.run_server(debug=True)
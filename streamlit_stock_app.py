import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from eodhd import APIClient
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import getpass
import os
import re
import datetime
import xgboost as xgb

# #Load Nvidia AI Endpoints API Key
# if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
#     nvapi_key = getpass.getpass("Enter your NVIDIA API key: ")
#     assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
#     os.environ["NVIDIA_API_KEY"] = nvapi_key

# Load Nvidia AI Endpoints API Key directly in program
os.environ["NVIDIA_API_KEY"] = "please enter your own api key"
assert os.environ["NVIDIA_API_KEY"].startswith("nvapi-"), "Invalid NVIDIA API key"

#Load Nvidia AI Endpoints LLM model 
llm = ChatNVIDIA(model="mistralai/mistral-large", max_tokens=1000) 

#Load EODHD API Key
api_key = 'please enter your own api key'
api = APIClient(api_key)

#Technical_Indicator
def Technical_Indicator(DF, a=12 ,b=26, c=9, n=14):
    """function to calculate MACD
       typical values a(fast moving average) = 12; 
                      b(slow moving average) =26; 
                      c(signal line ma window) =9"""
    df = DF.copy()
    df["ma_fast"] = df["Adj Close"].ewm(span=a, min_periods=a).mean()
    df["ma_slow"] = df["Adj Close"].ewm(span=b, min_periods=b).mean()
    df["macd"] = df["ma_fast"] - df["ma_slow"]
    df["signal"] = df["macd"].ewm(span=c, min_periods=c).mean()

    # function to calculate True Range and Average True Range
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Adj Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Adj Close"].shift(1))
    df["TR"] = df[["H-L","H-PC","L-PC"]].max(axis=1, skipna=False)
    df["ATR"] = df["TR"].ewm(com=n, min_periods=n).mean()

    # function to calculate RSI
    df["change"] = df["Adj Close"] - df["Adj Close"].shift(1)
    df["gain"] = np.where(df["change"]>=0, df["change"], 0)
    df["loss"] = np.where(df["change"]<0, -1*df["change"], 0)
    df["avgGain"] = df["gain"].ewm(alpha=1/n, min_periods=n).mean()
    df["avgLoss"] = df["loss"].ewm(alpha=1/n, min_periods=n).mean()
    df["rs"] = df["avgGain"]/df["avgLoss"]
    df["rsi"] = 100 - (100/ (1 + df["rs"]))

    # function to calculate ADX
    df["upmove"] = df["High"] - df["High"].shift(1)
    df["downmove"] = df["Low"].shift(1) - df["Low"]
    df["+dm"] = np.where((df["upmove"]>df["downmove"]) & (df["upmove"] >0), df["upmove"], 0)
    df["-dm"] = np.where((df["downmove"]>df["upmove"]) & (df["downmove"] >0), df["downmove"], 0)
    df["+di"] = 100 * (df["+dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["-di"] = 100 * (df["-dm"]/df["ATR"]).ewm(alpha=1/n, min_periods=n).mean()
    df["ADX"] = 100* abs((df["+di"] - df["-di"])/(df["+di"] + df["-di"])).ewm(alpha=1/n, min_periods=n).mean()

    # function to calculate Bollinger Band
    df["MB"] = df["Adj Close"].rolling(n).mean()
    df["UB"] = df["MB"] + 2*df["Adj Close"].rolling(n).std(ddof=0)
    df["LB"] = df["MB"] - 2*df["Adj Close"].rolling(n).std(ddof=0)
    df["BB_Width"] = df["UB"] - df["LB"]

    return df.loc[:,["macd","signal","ATR","rsi","ADX","MB","UB","LB","BB_Width"]]

# funtion to clean the textual data
def Clean_Text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()

# A function to count the number of tokens
def Count_Tokens(text):
    tokens = text.split()  
    return len(tokens)

template_Apple = """
        Identify the sentiment towards the Apple(AAPL) stocks of the news article from -10 to +10 where -10 being the most negative and +10 being the most positve , and 0 being neutral

        GIVE ANSWER IN ONLY ONE WORD AND THAT SHOULD BE THE SCORE

        Article : {statement}
        """

template_Nvidia = """
        Identify the sentiment towards the Nvidia(NVDA) stocks of the news article from -10 to +10 where -10 being the most negative and +10 being the most positve , and 0 being neutral

        GIVE ANSWER IN ONLY ONE WORD AND THAT SHOULD BE THE SCORE

        Article : {statement}
        """

template_Google = """
        Identify the sentiment towards the Google(GOOG) stocks of the news article from -10 to +10 where -10 being the most negative and +10 being the most positve , and 0 being neutral

        GIVE ANSWER IN ONLY ONE WORD AND THAT SHOULD BE THE SCORE

        Article : {statement}
        """

st.title('Stock Dashboard')
ticker = st.sidebar.selectbox('Ticker',("NVDA", "AAPL", "GOOG"))
start_date = 0
today = datetime.date.today()
# Format to 'YYYY-MM-DD'
end_date = today.strftime('%Y-%m-%d')

period = st.sidebar.selectbox('Period',("Week", "Month", "Trimester", "Year"))
# end_date = st.sidebar.date_input('End Date')
if period == "Week":
    one_week_ago = today - datetime.timedelta(days=7)
    start_date = one_week_ago.strftime('%Y-%m-%d')
elif period == "Month":
    one_month_ago = today - datetime.timedelta(days=30)
    start_date = one_month_ago.strftime('%Y-%m-%d')
elif period == "Trimester":
    one_trimester_ago = today - datetime.timedelta(days=90)
    start_date = one_trimester_ago.strftime('%Y-%m-%d')
elif period == "Year":
    one_year_ago = today - datetime.timedelta(days=365)
    start_date = one_year_ago.strftime('%Y-%m-%d')

# Drawing Stock Chart
data = yf.download(ticker, start=start_date, end=end_date)
fig = px.line(data, x = data.index, y = data['Adj Close'], title = ticker) 
st.plotly_chart(fig)

stock_information, technical_analysis, stock_sentiment = st.tabs(["Stock Information", "Technical Analysis", "Stock News Sentiment Prediction"])
with stock_information:
    st.header('Basic Stock Information',divider='grey')
    # Getting stock information
    stock = yf.Ticker(ticker)
    stock_info = stock.info
 
    market_cap = stock_info['marketCap']
    if market_cap >= 1_000_000_000_000:
        market_cap_str = f"{market_cap / 1_000_000_000_000:.2f} trillion USD"
    elif market_cap >= 1_000_000_000:
        market_cap_str = f"{market_cap / 1_000_000_000:.2f} billion USD"
    elif market_cap >= 1_000_000:
        market_cap_str = f"{market_cap / 1_000_000:.2f} million USD"
    else:
        market_cap_str = f"{market_cap:.2f} USD"

    pe_ratio = stock_info['trailingPE']
    pe_ratio_str = f"{pe_ratio:.2f} times"

    dividend_yield = stock_info.get('dividendYield', 0) * 100
    dividend_yield_str = f"{dividend_yield:.2f} %"

    st.html('<span class="column_indicator"></span>')
    with st.container():
        st.html('<span class="bottom_indicator"></span>')
        st.metric("Company Name", stock_info['longName'])
        st.metric("Industry", stock_info['industry'])
        st.metric("Market Cap", market_cap_str)
        st.metric("Earnings Per Share (EPS)", stock_info['trailingEps'])
        st.metric("Price to Earnings Ratio (P/E)", pe_ratio_str)
        st.metric("Dividend Yield", dividend_yield_str)

with technical_analysis:

    # showing technical indicator
    st.header(f'{ticker} Technical Indicator',divider='grey')
    ohlcv_data = yf.download(ticker,period='1mo',interval='15m')
    ohlcv_data.dropna(how="any",inplace=True)
    ohlcv_data[["macd","signal","ATR","rsi","ADX","MB","UB","LB","BB_Width"]] = Technical_Indicator(ohlcv_data, a=12 ,b=26, c=9)
    ohlcv_data = ohlcv_data.round(decimals=2)
    st.write(ohlcv_data[["macd","signal","ATR","rsi","ADX","MB","UB","LB","BB_Width"]])

    st.header(f'{ticker} Price Prediction',divider='grey')
    if (st.button("Run Price Prediction", type='primary')):
        with st.spinner('Processing...'):
            # downloading stock data
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=30)
            data = yf.download(ticker, start=start_date, end=end_date)

            # preprocessing the data
            data['Returns'] = data['Close'].pct_change()
            data = data.dropna()

            # train data and label
            lookback = 1
            X = data['Close'].shift(lookback).dropna().values.reshape(-1, 1)
            y = data['Close'][lookback:].values

            # split dataset
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # bulid and train XGBoost
            model = xgb.XGBRegressor(objective='reg:squarederror')
            model.fit(X_train, y_train)

            # Price prediction
            future_days = [1, 7, 30]
            predictions = {}
            for day in future_days:
                future_X = np.array([data['Close'].values[-1]]).reshape(-1, 1)
                for _ in range(day):
                    future_price = model.predict(future_X)
                    future_X = np.array([future_price]).reshape(-1, 1)
                predictions[day] = future_price[0]

            st.subheader("Prediction results:")
            st.html('<span class="column_indicator"></span>')
            with st.container():
                st.html('<span class="bottom_indicator"></span>')
                st.metric("Price tomorrow: ", predictions[1])
                st.metric("Price one week later: ", predictions[7])
                st.metric("Price one month later: ", predictions[30])
        st.success('finish!')
                
with stock_sentiment:
    st.header(f'{ticker} Sentiment Prediction')
    st.subheader('Predict with Nvidia AI Endpoints')
    st.write('LLM Model: mistral-large')

    if ticker == "NVDA":
        template = template_Nvidia
    elif ticker == "AAPL":
        template = template_Apple
    elif ticker == "GOOG":
        template = template_Google
    
    if (st.button("Run Sentiment Prediction", type='primary')):
        with st.spinner('Processing...'):
            resp = api.financial_news(s = ticker, from_date = start_date, to_date = end_date, limit = 10)
            df = pd.DataFrame(resp)
            
            # Filtering by token threshold
            df['content'] = df['content'].apply(Clean_Text)           
            df['TokenCount'] = df['content'].apply(Count_Tokens)
            token_count_threshold = 1000
            new_df = df[df['TokenCount'] < token_count_threshold]
            new_df = new_df.drop('TokenCount', axis = 1)
            new_df = new_df.reset_index(drop = True)

            # Langchain PromptTemplate 
            prompt = PromptTemplate(template = template, input_variables = ["statement"])
            llm_chain = LLMChain(prompt = prompt, llm = llm)
            
            x = []
            avg = 0
            count = 0
            for i in range(new_df.shape[0]):
                output = llm_chain.run(new_df['content'][i])
                print(f'News. {i}:')
                print(output)
                score = re.search(r"-?\d+", output).group()  # Sometime LLM will answer additional text, not a single number
                print(score)
                x.append(score)
                avg += int(score)
                count += 1

            print(x)
            avg = avg/count
            print(f'avg score: {avg}')

            series = pd.Series(x)
            # counting elments
            count_series = series.value_counts()
            # Changing the result to DataFrame
            df = count_series.reset_index()
            print(df)

            df.columns = ['index', 'nums']
            fig_pie = px.pie(df, values="nums", names="index", hole=0.5)
            fig_pie.update_traces(text = df["index"], textposition='outside')
            st.plotly_chart(fig_pie)
            st.subheader(f'Average Sentiment Score: {avg}')
        st.success('finish!')
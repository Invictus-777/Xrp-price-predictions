
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---- Load real XRP historical data ----
@st.cache_data
def load_data():
    df = yf.download('XRP-USD', period='5y', interval='1d')
    df.reset_index(inplace=True)
    return df

xrp_data = load_data()

# ---- Feature Engineering ----
xrp_data['Year'] = xrp_data['Date'].dt.year
xrp_data['Month'] = xrp_data['Date'].dt.month
xrp_data['BTC_Price'] = yf.download('BTC-USD', period='5y', interval='1d')['Close'].values
xrp_data.dropna(inplace=True)

xrp_data['BTC_Trend'] = np.where(xrp_data['BTC_Price'].diff() > 0, 'bull', 'bear')
xrp_data['Market_Sentiment'] = np.where(xrp_data['Close'].diff() > 0, 'positive', 'negative')
xrp_data['ODL_Growth'] = np.linspace(0, 5, len(xrp_data))
xrp_data['SEC_Status'] = 'none'

# ---- Model Training ----
features = ['Year', 'Month', 'BTC_Trend', 'Market_Sentiment', 'ODL_Growth']
target = 'Close'
categorical_features = ['BTC_Trend', 'Market_Sentiment']
numeric_features = ['Year', 'Month', 'ODL_Growth']

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X = xrp_data[features]
y = xrp_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---- Streamlit UI ----
st.title("📈 XRP Price Predictor & Tracker")

st.subheader("Live Market Data")
st.line_chart(xrp_data.set_index('Date')['Close'])

st.subheader("Predict Future Price")

year = st.slider("Year", int(xrp_data['Year'].min()), int(xrp_data['Year'].max()) + 1, int(xrp_data['Year'].max()))
month = st.slider("Month", 1, 12, 6)
btc_trend = st.selectbox("BTC Market Trend", ["bull", "bear"])
market_sentiment = st.selectbox("Market Sentiment", ["positive", "negative"])
odl_growth = st.slider("ODL Growth Stage", 0.0, 5.0, 3.0)

user_input = pd.DataFrame([{
    'Year': year,
    'Month': month,
    'BTC_Trend': btc_trend,
    'Market_Sentiment': market_sentiment,
    'ODL_Growth': odl_growth
}])

predicted_price = model.predict(user_input)[0]
st.success(f"Predicted XRP Price: ${predicted_price:.4f}")

# ---- Actual vs Predicted Plot ----
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_test['Year'] + X_test['Month'] / 12, y=y_test, mode='markers', name='Actual'))
fig.add_trace(go.Scatter(x=X_test['Year'] + X_test['Month'] / 12, y=y_pred, mode='markers', name='Predicted'))
fig.update_layout(title="Actual vs Predicted XRP Price", xaxis_title="Year.Month", yaxis_title="Price (USD)")
st.plotly_chart(fig)

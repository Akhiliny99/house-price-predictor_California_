

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap');

    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .title-text {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ade80, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle-text {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e2330;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        color: #e2e8f0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a2e1a, #1a1f2e);
        border: 2px solid #4ade80;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .prediction-label {
        color: #4ade80;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Syne', sans-serif;
        white-space: nowrap;
    }
    .prediction-range {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .insight-box {
        background: #13161e;
        border-left: 3px solid #60a5fa;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .stSlider > div > div { background: #1e2330; }
    div[data-testid="stSidebar"] { background: #13161e; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_artifacts():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_info.pkl', 'rb') as f:
        info = pickle.load(f)
    return model, scaler, info

model, scaler, model_info = load_artifacts()



st.markdown('<p class="title-text">🏡 California House Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">ML-powered prediction using XGBoost · 85.2% accuracy · Built with scikit-learn</p>', unsafe_allow_html=True)


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-label">Model</div><div class="metric-value">XGBoost</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value">0.852</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-label">Avg Error</div><div class="metric-value">$44K</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-label">Training Data</div><div class="metric-value">16,512</div></div>', unsafe_allow_html=True)

st.markdown("---")



st.sidebar.markdown("## 🎛️ House Features")
st.sidebar.markdown("Adjust the sliders to describe the house:")

st.sidebar.markdown("### 💰 Income & Economy")
MedInc = st.sidebar.slider(
    "Median Income (block group, $10Ks)",
    min_value=0.5, max_value=15.0, value=5.0, step=0.1,
    help="Median income of households in the block group"
)

st.sidebar.markdown("### 🏠 House Features")
HouseAge = st.sidebar.slider("House Age (years)", 1, 52, 20)
AveRooms = st.sidebar.slider("Average Rooms per House", 1.0, 15.0, 5.5, 0.1)
AveBedrms = st.sidebar.slider("Average Bedrooms per House", 0.5, 5.0, 1.1, 0.1)

st.sidebar.markdown("### 👥 Neighborhood")
Population = st.sidebar.slider("Block Group Population", 3, 5000, 1200)
AveOccup = st.sidebar.slider("Average Occupants per House", 1.0, 6.0, 2.8, 0.1)

st.sidebar.markdown("### 📍 Location (California)")
Latitude  = st.sidebar.slider("Latitude",  32.5, 42.0, 34.2, 0.01)
Longitude = st.sidebar.slider("Longitude", -124.5, -114.0, -118.5, 0.01)



rooms_per_person = AveRooms / AveOccup
bedroom_ratio    = AveBedrms / AveRooms
income_per_room  = MedInc / AveRooms
people_per_room  = Population / AveRooms


dist_sf = np.sqrt((Latitude - 37.77)**2 + (Longitude - (-122.42))**2)
dist_la = np.sqrt((Latitude - 34.05)**2 + (Longitude - (-118.24))**2)
dist_min_city = min(dist_sf, dist_la)


input_data = pd.DataFrame([[
    MedInc, HouseAge, AveRooms, AveBedrms,
    Population, AveOccup, Latitude, Longitude,
    rooms_per_person, bedroom_ratio, income_per_room, dist_min_city
]], columns=model_info['features'])


input_scaled = scaler.transform(input_data)



prediction = model.predict(input_scaled)[0]
prediction_dollars = prediction * 100_000
low  = (prediction - 0.44) * 100_000  
high = (prediction + 0.44) * 100_000



left_col, right_col = st.columns([1, 1.2], gap="large")

with left_col:
    
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Estimated House Price</div>
        <div class="prediction-value">${prediction_dollars:,.0f}</div>
        <div class="prediction-range">Likely range: ${max(0,low):,.0f} – ${high:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

   
    st.markdown("### 💡 What's driving this price?")

    if MedInc > 6:
        st.markdown('<div class="insight-box">📈 <b>High income area</b> — strong positive effect on price</div>', unsafe_allow_html=True)
    elif MedInc < 2:
        st.markdown('<div class="insight-box">📉 <b>Low income area</b> — negative effect on price</div>', unsafe_allow_html=True)

    if dist_min_city < 1.5:
        st.markdown('<div class="insight-box">🌆 <b>Near SF or LA</b> — proximity premium adds significant value</div>', unsafe_allow_html=True)
    elif dist_min_city > 5:
        st.markdown('<div class="insight-box">🌄 <b>Far from major cities</b> — lower location premium</div>', unsafe_allow_html=True)

    if rooms_per_person > 2.5:
        st.markdown('<div class="insight-box">🛋️ <b>Spacious per person</b> — positive signal for quality</div>', unsafe_allow_html=True)

    if HouseAge > 40:
        st.markdown('<div class="insight-box">🏚️ <b>Older home</b> — slight negative effect unless renovated</div>', unsafe_allow_html=True)

    
    st.markdown("### 📋 Your Input Summary")
    summary_df = pd.DataFrame({
        'Feature': ['Median Income', 'House Age', 'Avg Rooms', 'Population',
                    'Rooms/Person ✨', 'Income/Room ✨', 'Dist. to City ✨'],
        'Value': [f"${MedInc*10:.0f}K", f"{HouseAge} yrs", f"{AveRooms:.1f}",
                  f"{Population:,}", f"{rooms_per_person:.2f}",
                  f"{income_per_room:.2f}", f"{dist_min_city:.2f}°"]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


with right_col:
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction_dollars / 1000,
        number={'prefix': '$', 'suffix': 'K', 'font': {'size': 32}},
        delta={'reference': 207, 'suffix': 'K vs avg'},
        gauge={
            'axis': {'range': [0, 600], 'ticksuffix': 'K'},
            'bar': {'color': "#4ade80"},
            'bgcolor': "#1e2330",
            'bordercolor': "#2d3748",
            'steps': [
                {'range': [0, 150],   'color': '#1e2330'},
                {'range': [150, 300], 'color': '#1a2e1a'},
                {'range': [300, 450], 'color': '#1a3a1a'},
                {'range': [450, 600], 'color': '#1a4a1a'},
            ],
            'threshold': {
                'line': {'color': "#60a5fa", 'width': 3},
                'thickness': 0.8,
                'value': 207  
            }
        },
        title={'text': "Price vs California Average", 'font': {'color': '#94a3b8'}}
    ))
    fig_gauge.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        height=300,
        margin=dict(t=60, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    
    st.markdown("### 📊 Feature Importance")
    features_importance = {
        'income_per_room': 0.3154,
        'MedInc': 0.1972,
        'rooms_per_person': 0.1578,
        'Latitude': 0.0921,
        'Longitude': 0.0744,
        'dist_min_city': 0.0612,
        'AveOccup': 0.0389,
        'HouseAge': 0.0271,
        'AveRooms': 0.0198,
        'Population': 0.0087,
        'bedroom_ratio': 0.0052,
        'AveBedrms': 0.0022,
    }
    imp_df = pd.DataFrame(list(features_importance.items()),
                          columns=['Feature', 'Importance'])
    imp_df = imp_df.sort_values('Importance')
    imp_df['Color'] = imp_df['Feature'].apply(
        lambda x: '#4ade80' if x in ['income_per_room', 'rooms_per_person', 'dist_min_city', 'bedroom_ratio']
        else '#60a5fa'
    )

    fig_imp = go.Figure(go.Bar(
        x=imp_df['Importance'],
        y=imp_df['Feature'],
        orientation='h',
        marker_color=imp_df['Color'],
        text=[f"{v:.3f}" for v in imp_df['Importance']],
        textposition='outside'
    ))
    fig_imp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e2e8f0'},
        height=380,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis={'gridcolor': '#1e2330'},
        yaxis={'gridcolor': '#1e2330'}
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("🟢 Engineered features  🔵 Original features")



st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.8rem; padding: 1rem 0'>
    Built with ❤️ · XGBoost + Streamlit · California Housing Dataset ·
    R²=0.852 · <a href='https://github.com/Akhiliny99/house-price-predictor_California_' style='color:#4ade80'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)

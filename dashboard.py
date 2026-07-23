
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Konfigurasi Halaman
st.set_page_config(layout="wide", page_title="Dashboard MK", page_icon="⚖️")

# ============= SIDEBAR =============
with st.sidebar:
    st.title("⚖️ Dashboard MK")
    st.caption("Analisis Putusan Mahkamah Konstitusi")
    st.divider()
    
    data_source = st.radio("Pilih Sumber Data:", ["Upload File CSV", "Gunakan Data Default"])
    df = None
    
    if data_source == "Upload File CSV":
        uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(df)} baris data berhasil dimuat!")
    else:
        st.info("📊 Menggunakan data default...")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'tahun': np.random.choice([2018, 2019, 2020, 2021, 2022, 2023], n),
            'jenis_PHPU': np.random.choice([0,1], n, p=[0.7,0.3]),
            'jenis_Pengujian_UU': np.random.choice([0,1], n, p=[0.5,0.5]),
            'jenis_Sengketa_Lembaga': np.random.choice([0,1], n, p=[0.9,0.1]),
            'jenis_Lainnya': np.random.choice([0,1], n, p=[0.9,0.1]),
            'y_dikabulkan': np.random.choice([0,1], n, p=[0.7,0.3]),
            'kategori_amar': np.random.choice(['Dikabulkan', 'Ditolak', 'Tidak Diterima'], n, p=[0.3,0.4,0.3])
        })
        st.success(f"✅ {len(df)} baris data default siap!")

# ============= MAIN =============
if df is not None:
    df_clean = df.copy()
    
    # Deteksi kolom tahun
    if 'tahun' not in df_clean.columns:
        for col in df_clean.columns:
            if 'tahun' in col.lower():
                df_clean['tahun'] = pd.to_numeric(df_clean[col], errors='coerce')
                break
        if 'tahun' not in df_clean.columns:
            for col in df_clean.columns:
                if df_clean[col].dtype in ['int64', 'float64']:
                    if df_clean[col].min() >= 2000 and df_clean[col].max() <= 2030:
                        df_clean['tahun'] = df_clean[col]
                        break
    
    # Deteksi kolom amar
    if 'y_dikabulkan' not in df_clean.columns:
        if 'kategori_amar' in df_clean.columns:
            df_clean['y_dikabulkan'] = (df_clean['kategori_amar'] == 'Dikabulkan').astype(int)
        else:
            for col in df_clean.columns:
                if 'amar' in col.lower():
                    df_clean['y_dikabulkan'] = df_clean[col].astype(str).str.contains('dikabul', case=False).astype(int)
                    break
    
    df_clean = df_clean.dropna(subset=['tahun', 'y_dikabulkan'])
    df_clean['tahun'] = pd.to_numeric(df_clean['tahun'], errors='coerce').astype(int)
    df_clean['tahun_normalized'] = df_clean['tahun'] - 2000
    
    jenis_cols = [col for col in df_clean.columns if col.startswith('jenis_')]
    if not jenis_cols:
        df_clean['jenis_Pengujian_UU'] = np.random.choice([0,1], len(df_clean), p=[0.5,0.5])
        df_clean['jenis_PHPU'] = np.random.choice([0,1], len(df_clean), p=[0.7,0.3])
        jenis_cols = ['jenis_Pengujian_UU', 'jenis_PHPU']
    
    fitur_x = ['tahun_normalized'] + jenis_cols
    for f in fitur_x:
        if f in df_clean.columns:
            df_clean[f] = pd.to_numeric(df_clean[f], errors='coerce').fillna(0)
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan", "📈 Korelasi", "🧮 Model Regresi", "🔮 Prediksi"])
    
    with tab1:
        st.header("📊 Ringkasan Data")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Putusan", len(df_clean))
        with col2:
            dik = df_clean['y_dikabulkan'].sum()
            st.metric("Dikabulkan", f"{dik} ({dik/len(df_clean)*100:.1f}%)")
        with col3:
            st.metric("Rentang Tahun", f"{df_clean['tahun'].min()} - {df_clean['tahun'].max()}")
        with col4:
            st.metric("Jenis Perkara", len(jenis_cols))
        
        st.subheader("📋 Preview Data")
        st.dataframe(df_clean[['tahun', 'y_dikabulkan'] + jenis_cols].head(10))
        
        # Grafik Distribusi
        fig1, ax1 = plt.subplots()
        dist = df_clean['y_dikabulkan'].value_counts().sort_index()
        labels = ['Ditolak / Lainnya', 'Dikabulkan']
        ax1.bar(labels, dist.values, color=['#e74c3c', '#2ecc71'])
        ax1.set_ylabel('Jumlah')
        ax1.set_title('Distribusi Amar Putusan')
        st.pyplot(fig1)
    
    with tab2:
        st.header("📈 Analisis Korelasi")
        numeric_cols = ['tahun_normalized', 'y_dikabulkan'] + jenis_cols
        numeric_cols = [c for c in numeric_cols if c in df_clean.columns]
        
        if len(numeric_cols) > 1:
            corr_matrix = df_clean[numeric_cols].corr()
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax2)
            ax2.set_title('Heatmap Korelasi')
            st.pyplot(fig2)
    
    with tab3:
        st.header("🧮 Model Regresi Linier")
        X = df_clean[fitur_x].fillna(0)
        y = df_clean['y_dikabulkan']
        
        if len(X) > 5:
            test_size = st.slider("Ukuran Data Uji", 0.1, 0.4, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            col1, col2 = st.columns(2)
            col1.metric("R-Squared", f"{r2:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            
            st.subheader("Koefisien Regresi")
            coef_df = pd.DataFrame({'Fitur': fitur_x, 'Koefisien': model.coef_})
            st.dataframe(coef_df)
            
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['fitur_x'] = fitur_x
    
    with tab4:
        st.header("🔮 Prediksi")
        if 'model' in st.session_state:
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            fitur_x = st.session_state['fitur_x']
            
            tahun_input = st.number_input("Tahun", 2010, 2030, 2023)
            jenis_inputs = {}
            for f in fitur_x:
                if f != 'tahun_normalized':
                    jenis_inputs[f] = st.checkbox(f.replace('jenis_', ''))
            
            if st.button("Prediksi"):
                input_data = {'tahun_normalized': tahun_input - 2000}
                for f in fitur_x:
                    if f != 'tahun_normalized':
                        input_data[f] = 1 if jenis_inputs.get(f, False) else 0
                
                input_df = pd.DataFrame([input_data])[fitur_x].fillna(0)
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]
                pred = max(0, min(1, pred))
                
                st.success(f"Peluang Dikabulkan: {pred*100:.1f}%")
                st.progress(pred)

if __name__ == "__main__":
    st.caption("Dashboard MK - Final Project Big Data")

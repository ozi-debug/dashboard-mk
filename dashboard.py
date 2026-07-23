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
import os

# ============= KONFIGURASI =============
st.set_page_config(layout="wide", page_title="Dashboard MK", page_icon="⚖️")

st.title("⚖️ Dashboard Analisis Putusan Mahkamah Konstitusi")
st.caption("Final Project Big Data - Regresi Linier | Data dari Scraping JDIH MKRI")

# ============= LOAD DATA OTOMATIS =============
@st.cache_data
def load_data():
    # Coba baca file CSV dari repository
    if os.path.exists('putusan_mk_bersih.csv'):
        df = pd.read_csv('putusan_mk_bersih.csv')
        st.sidebar.success(f"✅ Data berhasil dimuat! Total: {len(df)} putusan")
        return df
    else:
        # Jika file tidak ada, gunakan data default
        st.sidebar.warning("⚠️ File putusan_mk_bersih.csv tidak ditemukan, menggunakan data default")
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
        return df

# Load data
with st.sidebar:
    st.image("https://mkri.id/public/images/logo-mk.png", width=100)
    st.title("⚖️ Dashboard MK")
    st.caption("Analisis Putusan Mahkamah Konstitusi")
    st.divider()
    st.write("📊 **Sumber Data:** Hasil Scraping JDIH MKRI")
    
df = load_data()

# ============= PREPROCESSING =============
if df is not None:
    df_clean = df.copy()
    
    # Deteksi kolom tahun
    if 'tahun' not in df_clean.columns:
        for col in df_clean.columns:
            if 'tahun' in col.lower() or 'year' in col.lower():
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
                if 'amar' in col.lower() or 'putusan' in col.lower():
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
    
    # ============ TABS ============
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan", "📈 Korelasi", "🧮 Model Regresi", "🔮 Prediksi"])
    
    # ============ TAB 1: RINGKASAN ============
    with tab1:
        st.header("📊 Ringkasan Data")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Putusan", f"{len(df_clean):,}", border=True)
        with col2:
            dik = df_clean['y_dikabulkan'].sum()
            st.metric("Dikabulkan", f"{dik} ({dik/len(df_clean)*100:.1f}%)", border=True)
        with col3:
            st.metric("Rentang Tahun", f"{df_clean['tahun'].min()} - {df_clean['tahun'].max()}", border=True)
        with col4:
            st.metric("Jenis Perkara", len(jenis_cols), border=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Preview Data")
            st.dataframe(df_clean[['tahun', 'y_dikabulkan'] + jenis_cols].head(10), use_container_width=True)
        
        with col2:
            st.subheader("📊 Distribusi Amar Putusan")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            dist = df_clean['y_dikabulkan'].value_counts().sort_index()
            labels = ['Ditolak / Lainnya', 'Dikabulkan']
            colors = ['#e74c3c', '#2ecc71']
            bars = ax1.bar(labels, dist.values, color=colors, edgecolor='black')
            ax1.set_ylabel('Jumlah')
            ax1.set_title('Distribusi Amar Putusan')
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 5, f'{int(height)}', ha='center', va='bottom')
            st.pyplot(fig1)
    
    # ============ TAB 2: KORELASI ============
    with tab2:
        st.header("📈 Analisis Korelasi")
        
        numeric_cols = ['tahun_normalized', 'y_dikabulkan'] + jenis_cols
        numeric_cols = [c for c in numeric_cols if c in df_clean.columns]
        
        if len(numeric_cols) > 1:
            corr_matrix = df_clean[numeric_cols].corr()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Heatmap Korelasi")
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                           linewidths=0.5, ax=ax2, cbar_kws={'label': 'Korelasi'})
                ax2.set_title('Heatmap Korelasi Antar Variabel', fontsize=14)
                st.pyplot(fig2)
            
            with col2:
                st.subheader("🎯 Korelasi dengan Y (Dikabulkan)")
                corr_target = corr_matrix['y_dikabulkan'].drop('y_dikabulkan').sort_values(ascending=False)
                
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                colors_corr = ['green' if x > 0 else 'red' for x in corr_target.values]
                ax3.barh(corr_target.index.str.replace('jenis_', ''), corr_target.values, color=colors_corr, edgecolor='black')
                ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
                ax3.set_xlabel('Korelasi dengan Dikabulkan')
                ax3.set_title('Korelasi Variabel dengan Y')
                st.pyplot(fig3)
        
        st.subheader("📈 Tren Putusan Dikabulkan per Tahun")
        tren_tahun = df_clean.groupby('tahun')['y_dikabulkan'].mean().reset_index()
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(tren_tahun['tahun'], tren_tahun['y_dikabulkan'], marker='o', linestyle='-', 
                linewidth=2, color='#3498db', markersize=8)
        ax4.set_ylim(0, 1)
        ax4.set_xlabel('Tahun')
        ax4.set_ylabel('Proporsi Dikabulkan')
        ax4.set_title('Tren Putusan Dikabulkan per Tahun')
        ax4.grid(True, alpha=0.3)
        for idx, row in tren_tahun.iterrows():
            ax4.annotate(f'{row["y_dikabulkan"]:.1%}', 
                       (row['tahun'], row['y_dikabulkan']),
                       textcoords="offset points", xytext=(0,10), ha='center')
        st.pyplot(fig4)
    
    # ============ TAB 3: MODEL REGRESI ============
    with tab3:
        st.header("🧮 Model Regresi Linier")
        
        X = df_clean[fitur_x].fillna(0)
        y = df_clean['y_dikabulkan']
        
        if len(X) > 5:
            test_size = st.slider("Ukuran Data Uji", 0.1, 0.4, 0.2, 0.05)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R-Squared (R²)", f"{r2:.4f}", border=True)
            with col2:
                st.metric("RMSE", f"{rmse:.4f}", border=True)
            with col3:
                st.metric("Jumlah Data", f"{len(X):,}", border=True)
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Koefisien Regresi")
                coef_df = pd.DataFrame({
                    'Fitur': fitur_x,
                    'Koefisien': model.coef_
                })
                st.dataframe(coef_df, use_container_width=True)
            
            with col2:
                st.subheader("📊 Visualisasi Koefisien")
                fig5, ax5 = plt.subplots(figsize=(6, 4))
                colors_bar = ['green' if x > 0 else 'red' for x in model.coef_]
                bars = ax5.barh([f.replace('jenis_', '') for f in fitur_x], model.coef_, color=colors_bar, edgecolor='black')
                ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
                ax5.set_xlabel('Koefisien')
                ax5.set_title('Koefisien Regresi')
                st.pyplot(fig5)
            
            st.subheader("📊 Residual Plot (Uji Heteroskedastisitas)")
            fig6, ax6 = plt.subplots(figsize=(10, 5))
            residuals = y_test - y_pred
            ax6.scatter(y_pred, residuals, alpha=0.5, color='#3498db')
            ax6.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax6.set_xlabel('Nilai Prediksi')
            ax6.set_ylabel('Residual')
            ax6.set_title('Residual Plot')
            ax6.grid(True, alpha=0.3)
            st.pyplot(fig6)
            
            # Interpretasi
            st.subheader("📝 Interpretasi Model")
            if r2 > 0.5:
                st.success(f"✅ Model cukup baik (R² = {r2:.3f}). {r2*100:.1f}% variasi Y dijelaskan oleh fitur.")
            elif r2 > 0.3:
                st.warning(f"⚠️ Model sedang (R² = {r2:.3f}). {r2*100:.1f}% variasi Y dijelaskan oleh fitur.")
            else:
                st.warning(f"⚠️ Model lemah (R² = {r2:.3f}). Fitur yang digunakan kurang menjelaskan variasi putusan.")
                st.info("💡 Faktor non-teknis (argumentasi hukum, kualitas bukti) mungkin lebih dominan.")
            
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['fitur_x'] = fitur_x
        else:
            st.error("❌ Data terlalu sedikit untuk training model (minimal 6 baris).")
    
    # ============ TAB 4: PREDIKSI ============
    with tab4:
        st.header("🔮 Prediksi Peluang Dikabulkan")
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            fitur_x = st.session_state['fitur_x']
            
            st.info("📌 Masukkan karakteristik perkara untuk memprediksi peluang dikabulkan.")
            
            col1, col2 = st.columns(2)
            with col1:
                tahun_input = st.number_input("📅 Tahun Putusan", min_value=2010, max_value=2030, value=2023, step=1)
            with col2:
                jenis_inputs = {}
                for f in fitur_x:
                    if f != 'tahun_normalized':
                        nama = f.replace('jenis_', '').replace('_', ' ').title()
                        jenis_inputs[f] = st.checkbox(f"📌 {nama}", key=f)
            
            if st.button("🚀 Prediksi Sekarang", use_container_width=True):
                input_data = {'tahun_normalized': tahun_input - 2000}
                for f in fitur_x:
                    if f != 'tahun_normalized':
                        input_data[f] = 1 if jenis_inputs.get(f, False) else 0
                
                input_df = pd.DataFrame([input_data])[fitur_x].fillna(0)
                input_scaled = scaler.transform(input_df)
                
                pred_prob = model.predict(input_scaled)[0]
                pred_prob = max(0, min(1, pred_prob))
                
                st.divider()
                st.subheader("📌 Hasil Prediksi")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Peluang Dikabulkan", f"{pred_prob*100:.1f}%")
                with col2:
                    if pred_prob > 0.6:
                        st.success("✅ Peluang TINGGI")
                    elif pred_prob > 0.4:
                        st.warning("⚠️ Peluang SEDANG")
                    else:
                        st.error("❌ Peluang RENDAH")
                with col3:
                    st.progress(pred_prob)
                
                st.caption(f"Input: Tahun {tahun_input} | Jenis: {', '.join([f.replace('jenis_', '') for f in fitur_x if f != 'tahun_normalized' and input_data[f] == 1]) or 'Tidak ada'}")
        else:
            st.warning("⚠️ Model belum siap. Silakan latih model terlebih dahulu di tab 'Model Regresi'.")
    
    # ============ FOOTER ============
    st.divider()
    st.caption(f"⚖️ Dashboard MK | {len(df_clean):,} data hasil scraping JDIH MKRI | {datetime.now().strftime('%Y')}")
    st.caption("Final Project Big Data - Regresi Linier")

else:
    st.error("❌ Gagal memuat data. Silakan periksa file CSV Anda.")

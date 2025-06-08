import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from streamlit_echarts import st_echarts
import io

# --- Fungsi Prediksi ---
def predict_dropout_risk(data):
    df_input = pd.DataFrame([data])

    df_input = df_input.rename(columns={
        "IPK Semester 1": "IPK_Semester_1",
        "IPK Semester 2": "IPK_Semester_2",
        "IPK Semester 3": "IPK_Semester_3",
        "IPK Semester 4": "IPK_Semester_4",
        "IPK Semester 5": "IPK_Semester_5",
        "IPK Semester 6": "IPK_Semester_6",
        "Kehadiran per Mata Kuliah (%)": "Kehadiran_Per_Mata_Kuliah",
        "Riwayat Pengambilan Ulang (Jumlah)": "Riwayat_Pengambilan_Ulang",
        "Aktivitas Sistem Pembelajaran Daring (Skor)": "Aktivitas_Sistem_Pembelajaran_Daring",
        "Beban Kerja (Jam/Minggu)": "Beban_Kerja_JamPerMinggu",
        "Status Pekerjaan": "Status_Pekerjaan_Label"
    })

    df_input['Status_Pekerjaan_Label'] = df_input['Status_Pekerjaan_Label'].apply(lambda x: 0 if x == 'Bekerja' else 1)

    kolom_training = [
        'IPK_Semester_1', 'IPK_Semester_2', 'IPK_Semester_3',
        'IPK_Semester_4', 'IPK_Semester_5', 'IPK_Semester_6',
        'Kehadiran_Per_Mata_Kuliah', 'Riwayat_Pengambilan_Ulang',
        'Aktivitas_Sistem_Pembelajaran_Daring', 'Beban_Kerja_JamPerMinggu',
        'Status_Pekerjaan_Label'
    ]
    df_input = df_input[kolom_training]

    input_scaled = scaler.transform(df_input)
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    return prediction[0], prediction_proba

# --- Fungsi Visualisasi Gauge ---
def tampilkan_gauge(probabilitas, aman=True):
    option = {
        "series": [
            {
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "min": 0,
                "max": 100,
                "progress": {"show": True, "width": 18},
                "axisLine": {"lineStyle": {"width": 18}},
                "pointer": {"show": True},
                "title": {"show": True, "offsetCenter": [0, "70%"]},
                "detail": {
                    "valueAnimation": True,
                    "formatter": "{value}%",
                    "offsetCenter": [0, "40%"],
                    "fontSize": 24,
                    "color": "#FF4B4B" if not aman else "#3EC70B"
                },
                "data": [{"value": probabilitas, "name": "Risiko"}],
            }
        ]
    }
    st_echarts(options=option, height="300px")

# --- Fungsi Visualisasi IPK ---
def tampilkan_ipk_chart(data):
    ipk_dict = {
        'Semester 1': data["IPK Semester 1"],
        'Semester 2': data["IPK Semester 2"],
        'Semester 3': data["IPK Semester 3"],
        'Semester 4': data["IPK Semester 4"],
        'Semester 5': data["IPK Semester 5"],
        'Semester 6': data["IPK Semester 6"]
    }
    df_ipk = pd.DataFrame(list(ipk_dict.items()), columns=['Semester', 'IPK'])
    fig = px.line(df_ipk, x='Semester', y='IPK', title='📈 Perkembangan IPK per Semester', markers=True)
    st.plotly_chart(fig, use_container_width=True)

# --- Insight Otomatis ---
def insight_analisis(data):
    st.subheader("📌 Insight Otomatis")
    if data['IPK Semester 6'] < 2.5:
        st.markdown("- IPK semester akhir cukup rendah, ini sinyal lampu kuning! 🚨")
    if data['Kehadiran per Mata Kuliah (%)'] < 75:
        st.markdown("- Kehadiran di bawah 75%, perlu ditingkatkan biar gak ketinggalan materi.")
    if data['Riwayat Pengambilan Ulang (Jumlah)'] >= 3:
        st.markdown("- Banyak ambil ulang matkul, bisa jadi warning soal kesulitan akademik.")
    if data['Status Pekerjaan'] == 'Bekerja' and data['Beban Kerja (Jam/Minggu)'] > 20:
        st.markdown("- Beban kerja tinggi + kuliah? Bisa pengaruh banget ke performa akademik.")

# --- Halaman Utama ---
st.set_page_config(page_title="Prediksi Risiko Drop Out", layout="wide")

st.sidebar.title("ℹ️ Informasi Model")
st.sidebar.info(
    "Model ini membantu identifikasi mahasiswa berisiko drop out secara dini."
)
st.sidebar.success("🚀 Powered by Random Forest & Streamlit")
st.sidebar.markdown("---")
st.sidebar.header("Anggota Kelompok")
st.sidebar.markdown("""
- **Neli Agustin** (G1A022048)
- **Rizki Ramadani Dalimunthe** (G1A022054)
- **Yuda Reyvandra Herman** (G1A022072)
""")
# --- Sticky Watermark di Sidebar Bawah Banget ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]::after {
        content: "© 2025 Team 9 - Data Mining";
        position: absolute;
        bottom: 10px;
        left: 90px;
        font-size: 13px;
        color: #999999;
        opacity: 0.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🎓📊 Sistem Early Warning Drop Out Mahasiswa")
st.subheader("✨ Yuk, cari tahu seberapa besar kemungkinan DO mahasiswa berdasarkan datanya.")

try:
    model = joblib.load('rf_model_best.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Upload 'rf_model_best.pkl' dan 'scaler.pkl'.")
    st.stop()

# --- Input Pengguna ---
def user_input_features():
    st.header("📝 Masukkan Data Mahasiswa")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Data Akademik")
        ipk1 = st.number_input('IPK Semester 1', 0.0, 4.0, 3.0, 0.01)
        ipk2 = st.number_input('IPK Semester 2', 0.0, 4.0, 3.0, 0.01)
        ipk3 = st.number_input('IPK Semester 3', 0.0, 4.0, 2.8, 0.01)
        ipk4 = st.number_input('IPK Semester 4', 0.0, 4.0, 2.7, 0.01)
        ipk5 = st.number_input('IPK Semester 5', 0.0, 4.0, 2.5, 0.01)
        ipk6 = st.number_input('IPK Semester 6', 0.0, 4.0, 2.3, 0.01)

    with col2:
        st.subheader("🌐 Data Aktivitas")
        kehadiran = st.slider('Kehadiran per Mata Kuliah (%)', 0, 100, 80)
        ulang = st.number_input('Riwayat Pengambilan Ulang (Jumlah)', 0, 10, 2)
        aktivitas_daring = st.slider('Aktivitas Sistem Pembelajaran Daring (Skor)', 0, 100, 50)
        st.subheader("💼 Data Non-Akademik")
        status_kerja = st.selectbox('Status Pekerjaan', ('Tidak Bekerja', 'Bekerja'))
        beban_kerja = 0
        if status_kerja == 'Bekerja':
            beban_kerja = st.slider('Beban Kerja (Jam/Minggu)', 0, 60, 20)

    return {
        "IPK Semester 1": ipk1,
        "IPK Semester 2": ipk2,
        "IPK Semester 3": ipk3,
        "IPK Semester 4": ipk4,
        "IPK Semester 5": ipk5,
        "IPK Semester 6": ipk6,
        "Kehadiran per Mata Kuliah (%)": kehadiran,
        "Riwayat Pengambilan Ulang (Jumlah)": ulang,
        "Aktivitas Sistem Pembelajaran Daring (Skor)": aktivitas_daring,
        "Status Pekerjaan": status_kerja,
        "Beban Kerja (Jam/Minggu)": beban_kerja,
    }

input_data = user_input_features()
st.markdown("---")

# --- Tombol Prediksi ---
if st.button('🚀 Prediksi Risiko', use_container_width=True):
    prediksi, proba = predict_dropout_risk(input_data)

    st.header('🔍 Hasil Prediksi')
    tampilkan_gauge(proba[0][1]*100, aman=(prediksi == 0))

    if prediksi == 1:
        st.error('**Status: RISIKO TINGGI**')
        st.warning(f"Probabilitas berisiko tinggi: **{proba[0][1]*100:.2f}%**")
        st.markdown("**Rekomendasi:** Segera lakukan konseling akademik dan evaluasi beban studi.")
    else:
        st.success('**Status: AMAN**')
        st.info(f"Probabilitas untuk tetap aman: **{proba[0][0]*100:.2f}%**")
        st.markdown("**Rekomendasi:** Tetap dipantau dan diberi dukungan akademik yang konsisten.")

    with st.expander("📋 Lihat Ringkasan Data Mahasiswa"):
        df_tampil = pd.DataFrame([input_data]).T
        df_tampil.columns = ["Nilai"]
        st.table(df_tampil)

    tampilkan_ipk_chart(input_data)
    insight_analisis(input_data)

    # Download CSV
    csv = pd.DataFrame([input_data]).assign(Hasil=('Risiko Tinggi' if prediksi else 'Aman')).to_csv(index=False)
    st.download_button("📄 Unduh Hasil Prediksi (CSV)", data=csv, file_name="hasil_prediksi.csv", mime='text/csv')

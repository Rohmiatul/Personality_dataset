import streamlit as st
import pandas as pd
import joblib # Asumsikan Anda menyimpan model imputer dan KNN menggunakan joblib atau pickle

# --- Muat Model dan Imputer ---
# Anda perlu memastikan 'imputer.joblib' dan 'knn_model.joblib' berada di direktori yang sama
# dengan skrip aplikasi Streamlit Anda saat Anda deploy ke Streamlit Cloud,
# atau berikan path yang benar jika mereka berada di subdirektori.
try:
    imputer = joblib.load('imputer.joblib')
    knn = joblib.load('knn_model.joblib')
except FileNotFoundError:
    st.error("Error: File model atau imputer tidak ditemukan. Pastikan 'imputer.joblib' dan 'knn_model.joblib' berada di direktori yang benar.")
    st.stop() # Hentikan aplikasi jika file penting hilang
except Exception as e:
    st.error(f"Error memuat model atau imputer: {e}")
    st.stop()

# --- Judul dan Deskripsi Aplikasi Streamlit ---
st.set_page_config(page_title="Aplikasi Prediksi Kepribadian", layout="centered")
st.title("Aplikasi Prediksi Kepribadian")
st.write("Aplikasi ini memprediksi kepribadian (Ekstrovert/Introvert) berdasarkan aktivitas sosial Anda.")

# --- Kolom Input ---
st.header("Masukkan Data Anda")

col1, col2, col3 = st.columns(3)

with col1:
    new_Social_event_attendance = st.number_input(
        "Jumlah Kegiatan sosial:",
        min_value=0.0,
        help="Jumlah rata-rata kegiatan sosial yang Anda ikuti per bulan."
    )
with col2:
    new_Going_outside = st.number_input(
        "Jumlah Kegiatan bermain di luar:",
        min_value=0.0,
        help="Jumlah rata-rata waktu yang Anda habiskan di luar rumah per minggu."
    )
with col3:
    new_Friends_circle_size = st.number_input(
        "Ukuran Lingkaran Pertemanan:",
        min_value=0.0,
        help="Perkiraan jumlah teman dekat yang Anda miliki."
    )

# --- Tombol Prediksi ---
if st.button("Prediksi Kepribadian"):
    try:
        # Buat DataFrame dari input baru
        new_data_df = pd.DataFrame(
            [[new_Social_event_attendance, new_Going_outside, new_Friends_circle_size]],
            columns=['Social_event_attendance', 'Going_outside', 'Friends_circle_size']
        )

        # Imputasi data baru (asumsikan imputer telah di-fit pada data pelatihan)
        new_data_df_imputed = imputer.transform(new_data_df)
        new_data_df_imputed = pd.DataFrame(new_data_df_imputed, columns=new_data_df.columns)

        # Lakukan prediksi
        predicted_code = knn.predict(new_data_df_imputed)[0]

        # Konversi prediksi ke label
        label_mapping = {1: 'Ekstrovert', 0: 'Introvert'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        # Tampilkan hasil
        st.success("### Hasil Prediksi:")
        st.write(f"Berdasarkan data yang Anda masukkan, prediksi kepribadian Anda adalah: **{predicted_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
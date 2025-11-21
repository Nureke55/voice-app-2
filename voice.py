import streamlit as st
import numpy as np
import wave
import scipy.signal as signal
import os
import pickle

st.set_page_config(page_title="Voice Access", page_icon="🔐")
st.title("🔐 Тек бір дауысқа рұқсат жүйесі")

DATA_FILE = "voice_profile.pkl"

def extract_voice_features(file):
    with wave.open(file, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)

    freq, time, spec = signal.spectrogram(audio)
    return np.mean(spec, axis=1)

# ================== ЭТАЛОН ДАУЫС ТІРКЕУ ==================
st.subheader("1) ӨЗ ДАУЫСЫҢДЫ ТІРКЕУ (бір рет)")

ref_audio = st.file_uploader("Өзіңнің WAV дауысыңды жүкте", type=["wav"])

if ref_audio and not os.path.exists(DATA_FILE):
    with open("ref.wav", "wb") as f:
        f.write(ref_audio.read())

    features = extract_voice_features("ref.wav")

    with open(DATA_FILE, "wb") as f:
        pickle.dump(features, f)

    st.success("✅ Сенің дауысың сақталды! Енді тек осы дауыс өтеді.")

elif os.path.exists(DATA_FILE):
    st.info("ℹ️ Эталон дауыс бұрын сақталған.")

# ================== ДАУЫС ТЕКСЕРУ ==================
st.subheader("2) ДАУЫС АРҚЫЛЫ КІРУ")

login_audio = st.file_uploader("Дауыс арқылы кіру (WAV)", type=["wav"], key="login")

if login_audio and os.path.exists(DATA_FILE):
    with open("test.wav", "wb") as f:
        f.write(login_audio.read())

    test_features = extract_voice_features("test.wav")

    with open(DATA_FILE, "rb") as f:
        reference_features = pickle.load(f)

    similarity = np.corrcoef(reference_features, test_features)[0][1]

    st.write("Ұқсастық деңгейі:", round(float(similarity), 2))

    # ҚАТАҢ ПОРОГ
    if similarity > 0.85:
        st.success("✅ Доступ разрешён — бұл сіздің дауысыңыз")
    else:
        st.error("❌ Доступ запрещён — бөтен дауыс анықталды!")

elif login_audio and not os.path.exists(DATA_FILE):
    st.warning("Алдымен өз дауысыңды тірке!")

# ================== ҚАЛПЫНА КЕЛТІРУ ==================
st.subheader("🔄 Дауысты қайта тіркеу")
if st.button("Эталонды өшіру"):
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        st.success("Эталон өшірілді. Қайта тіркеуге болады.")

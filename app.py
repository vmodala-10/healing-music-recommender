import streamlit as st
import joblib
import pandas as pd
import os

# Load model and encoders
model = joblib.load("ensemble_model.pkl")
encoders = joblib.load("label_encoders.pkl")
df = pd.read_csv("healing_music_dataset.csv")

feedback_file = "feedback.csv"

# Predict raga
def predict_raga(pain, tempo, pitch, mood):
    input_df = pd.DataFrame([[pain, tempo, pitch, mood]], columns=["Pain", "Tempo", "Pitch", "Mood"])
    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])
    pred = model.predict(input_df)[0]
    raga = encoders["Raga"].inverse_transform([pred])[0]
    return raga

# Play raga audio
def play_raga_audio(raga_name):
    file_path = os.path.join("audio samples", f"{raga_name.lower()}.mp3")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
    else:
        st.error("Audio sample not found.")

# Save feedback
def save_feedback(pain, tempo, pitch, mood, raga, feedback_text):
    new_data = {
        "Pain": pain,
        "Tempo": tempo,
        "Pitch": pitch,
        "Mood": mood,
        "Raga": raga,
        "Feedback": feedback_text
    }

    try:
        file_exists = os.path.isfile(feedback_file)
        with open(feedback_file, 'a') as f:
            if not file_exists:
                f.write(",".join(new_data.keys()) + "\n")
            f.write(",".join(map(str, new_data.values())) + "\n")

        st.success("✅ Feedback saved successfully.")

    except Exception as e:
        st.error("⚠️ Failed to save feedback.")
        st.write(str(e))

# UI
st.title("🎶 Healing Music Recommender")
option = st.radio("Choose an action:", ["Get Music Recommendation", "Explore Ragas"])

if option == "Get Music Recommendation":
    st.subheader("Tell us how you're feeling:")
    pain = st.selectbox("Select Pain Type", list(encoders["Pain"].classes_))
    tempo = st.selectbox("Select Tempo", list(encoders["Tempo"].classes_))
    pitch = st.selectbox("Select Pitch", list(encoders["Pitch"].classes_))
    mood = st.selectbox("Select Mood", list(encoders["Mood"].classes_))

    if st.button("Recommend"):
        raga = predict_raga(pain, tempo, pitch, mood)
        st.success(f"🎵 Recommended Raga: {raga}")

        # Audio player
        play_raga_audio(raga)

        st.markdown("### Was this helpful?")
        col1, col2 = st.columns(2)
        if col1.button("👍 Helpful"):
            save_feedback(pain, tempo, pitch, mood, raga, "Helpful")
        if col2.button("👎 Not Helpful"):
            save_feedback(pain, tempo, pitch, mood, raga, "Not Helpful")

elif option == "Explore Ragas":
    st.subheader("🔍 Filter Ragas")
    pain_filter = st.selectbox("Pain", list(encoders["Pain"].classes_))
    mood_filter = st.selectbox("Mood", list(encoders["Mood"].classes_))

    result = df[(df["Pain"].str.lower() == pain_filter.lower()) & (df["Mood"].str.lower() == mood_filter.lower())]
    if not result.empty:
        st.write("🎼 Matching Ragas:")
        for r in result["Raga"].unique():
            with st.expander(f"🎵 {r}"):
                st.write(f"🎶 Listen to {r}:")
                file_path = os.path.join("audio samples", f"{r.lower()}.mp3")
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")

                    st.download_button(
                        label=f"⬇️ Download {r}",
                        data=audio_bytes,
                        file_name=f"{r.lower()}.mp3",
                        mime="audio/mp3"
                    )
                else:
                    st.warning("⚠️ Audio sample not found.")
    else:
        st.warning("No ragas found for the selected filters.")

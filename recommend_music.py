import joblib
import pandas as pd
import pygame
import time
import os

pygame.mixer.init()

# Load model and encoders
model = joblib.load("ensemble_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Show available options from encoders
def show_options():
    print("🎯 Available Options:")
    for col, le in encoders.items():
        if col != "Raga":
            print(f"{col}: {list(le.classes_)}")

# Predict Raga
def predict_raga(pain, tempo, pitch, mood):
    input_df = pd.DataFrame([[pain, tempo, pitch, mood]], columns=["Pain", "Tempo", "Pitch", "Mood"])
    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])
    pred = model.predict(input_df)[0]
    raga = encoders["Raga"].inverse_transform([pred])[0]
    return raga

# Get valid input from user
def get_valid_input(prompt, options):
    while True:
        user_input = input(prompt).strip().title()
        if user_input in options:
            return user_input
        else:
            print(f"❌ Invalid input. Please choose from: {options}")

# Load your dataset for exploration
df = pd.read_csv("healing_music_dataset.csv")

# Explore matching ragas based on pain and mood filters
def explore_ragas(df, pain_filter, mood_filter):
    result = df[(df["Pain"].str.lower() == pain_filter.lower()) &
                (df["Mood"].str.lower() == mood_filter.lower())]

    if not result.empty:
        print("\n🎼 Ragas matching your filter:")
        for raga in result["Raga"].unique():
            print(f"- {raga}")
    else:
        print("😕 No ragas found for the selected criteria.")

def play_raga_audio(raga_name):
    file_path = os.path.join("audio samples", f"{raga_name.lower()}.mp3")
    if os.path.exists(file_path):
        print(f"🔊 Playing a sample of Raga {raga_name}...")
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        time.sleep(10)  # Adjust duration as needed
        pygame.mixer.music.stop()
    else:
        print(f"❌ Audio sample not found: {file_path}")
        
def play_multiple_ragas(df, mood_filter, pain_filter=None):
    if pain_filter:
        filtered = df[(df["Mood"].str.lower() == mood_filter.lower()) &
                      (df["Pain"].str.lower() == pain_filter.lower())]
    else:
        filtered = df[df["Mood"].str.lower() == mood_filter.lower()]
    
    ragas = filtered["Raga"].unique()
    
    if ragas.any():
        print("\n🧘 Playing ragas for your mood:")
        for raga in ragas:
            print(f"- {raga}")
            play_raga_audio(raga)
            time.sleep(2)  # short pause between ragas
    else:
        print("😕 No ragas found for the selected mood and pain filter.")
    

            

# --- Main script loop ---
print("🎶 Welcome to the Healing Music Recommender 🎶")

while True:
    print("\nWhat would you like to do?")
    print("1. Get a music recommendation")
    print("2. Explore ragas based on pain and mood")
    print("3. Hear multiple ragas for your mood")

    choice = input("Enter 1, 2 or 3: ").strip()

    if choice == "1":
        show_options()
        pain = get_valid_input("\nEnter your Pain type: ", list(encoders["Pain"].classes_))
        tempo = get_valid_input("Enter Tempo: ", list(encoders["Tempo"].classes_))
        pitch = get_valid_input("Enter Pitch: ", list(encoders["Pitch"].classes_))
        mood = get_valid_input("Enter Mood: ", list(encoders["Mood"].classes_))

        try:
            recommended_raga = predict_raga(pain, tempo, pitch, mood)
            print(f"\n🎵 Based on your inputs, we recommend the Raga: **{recommended_raga}**")
            


            # Ask the user if they want to listen
            listen = input("🎧 Would you like to listen to a sample of this Raga? (yes/no): ").strip().lower()
            if listen in ["yes", "y"]:
                play_raga_audio(recommended_raga)
            else:
                print("👍 Okay! Hope the recommendation helps.")
        except Exception as e:
            print("\n⚠️ Unexpected error occurred. Please try again.")

    elif choice == "2":
        show_options()
        pain_filter = get_valid_input("Enter Pain type to filter: ", list(encoders["Pain"].classes_))
        mood_filter = get_valid_input("Enter Mood to filter: ", list(encoders["Mood"].classes_))
        explore_ragas(df, pain_filter, mood_filter)
        
    elif choice == "3":
        show_options()
        mood_filter = get_valid_input("Enter Mood: ", list(encoders["Mood"].classes_))
        pain_filter = input("Optional: Enter Pain type to narrow down (or press Enter to skip): ").strip().title()
        if pain_filter in list(encoders["Pain"].classes_):
            play_multiple_ragas(df, mood_filter, pain_filter)
        else:
            play_multiple_ragas(df, mood_filter)

        
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

    try_again = input("\n🔁 Would you like to continue? (yes/no): ").strip().lower()
    if try_again not in ['yes', 'y']:
        print("\n🙏 Thank you for using the Healing Music Recommender. Stay healthy and peaceful!")
        break
    
def save_feedback(feedback_file, pain, tempo, pitch, mood, raga, feedback_text):
    new_data = {
        "Pain": pain,
        "Tempo": tempo,
        "Pitch": pitch,
        "Mood": mood,
        "Raga": raga,
        "Feedback": feedback_text
    }
    file_exists = os.path.isfile(feedback_file)
    with open(feedback_file, 'a') as f:
        if not file_exists:
            f.write(",".join(new_data.keys()) + "\n")
        f.write(",".join(map(str, new_data.values())) + "\n")


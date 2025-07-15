from playsound import playsound

try:
    playsound('beep.wav')
    print("Beep played successfully!")
except Exception as e:
    print(f"Error playing sound: {e}")
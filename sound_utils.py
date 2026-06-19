import os
import wave
import struct
import numpy as np

def generate_chime_wav(filename="notification.wav", sample_rate=44100):
    """
    Generates a pleasant double-tone chime and saves it as a WAV file.
    Tone 1: 600Hz for 0.15 seconds.
    Silence: 0.05 seconds.
    Tone 2: 850Hz for 0.30 seconds.
    Includes fade-out to prevent audio pops/clicks.
    """
    try:
        # Tone 1: 0.15s of 600Hz
        t1 = np.linspace(0, 0.15, int(sample_rate * 0.15), endpoint=False)
        wave1 = np.sin(2 * np.pi * 600.0 * t1)
        # Fade out last 20%
        fade_len1 = int(len(wave1) * 0.2)
        wave1[-fade_len1:] *= np.linspace(1, 0, fade_len1)
        
        # Silence gap: 0.05s
        silence = np.zeros(int(sample_rate * 0.05))
        
        # Tone 2: 0.3s of 850Hz
        t2 = np.linspace(0, 0.3, int(sample_rate * 0.3), endpoint=False)
        wave2 = np.sin(2 * np.pi * 850.0 * t2)
        # Fade out last 20%
        fade_len2 = int(len(wave2) * 0.2)
        wave2[-fade_len2:] *= np.linspace(1, 0, fade_len2)
        
        # Combine the signals
        audio_data = np.concatenate([wave1, silence, wave2])
        
        # Normalize to 16-bit PCM scale (-32768 to 32767)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Ensure output directory exists
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        with wave.open(filename, 'wb') as w:
            w.setnchannels(1)      # Mono
            w.setsampwidth(2)      # 2 bytes (16-bit)
            w.setframerate(sample_rate)
            w.writeframes(audio_data.tobytes())
        print(f"Successfully generated notification sound: {filename}")
        return True
    except Exception as e:
        print(f"Error generating WAV file: {e}")
        return False

if __name__ == "__main__":
    # If run directly, create the notification sound in the local directory
    generate_chime_wav("notification.wav")

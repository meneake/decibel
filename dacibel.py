import librosa
from moviepy.editor import AudioFileClip
from transformers import pipeline
from reazonspeech.espnet.asr import transcribe, audio_from_path, load_model

class Decibel:

    def __init__(self, source, is_Decibel=True):
        self.source = source
        self.is_Decibel = is_Decibel
        if self.is_Decibel:
            self.process_audio()

    def process_audio(self):
        # Extract audio from video
        audio = AudioFileClip(self.source)
        wav_path = self.source.replace('.mp4', '.wav')
        audio.write_audiofile(wav_path)

        # Load audio file and calculate RMS and Decibel
        y, sr = librosa.load(wav_path)
        rms = librosa.feature.rms(y=y)
        decibel = librosa.amplitude_to_db(rms)
        time = librosa.times_like(decibel, sr=sr)

        # Initialize variables
        time_fromStart_db = 0
        time_num = 0
        decibel_at_this_second = 0
        decibel_max_at_this_second = -100
        decibel_per_second = []
        decibel_max_per_second = []

        # Calculate Decibel per second
        for i, t in enumerate(time):
            if time_fromStart_db <= t < time_fromStart_db + 1:
                decibel_at_this_second += decibel[0][i]
                if decibel[0][i] > decibel_max_at_this_second:
                    decibel_max_at_this_second = decibel[0][i]
                time_num += 1
            else:
                decibel_ave_at_this_second = decibel_at_this_second / time_num
                decibel_per_second.append(decibel_ave_at_this_second)
                decibel_max_per_second.append(decibel_max_at_this_second)
                decibel_at_this_second = decibel[0][i]
                time_num = 1
                decibel_max_at_this_second = decibel[0][i]
                time_fromStart_db += 1

        # Cut the audio file into smaller pieces for emotion recognition
        wave_file_list = self.cut_wav(wav_path, 5)
        emotion_list = []
        emotion_labels = ["happy", "sad", "angry", "neutral"]  # Example emotion labels
        emotion_per_second = {label: [] for label in emotion_labels}

        # Perform emotion recognition on each segment
        inference_pipeline = pipeline(task="automatic-speech-recognition", model="iic/emotion2vec_base_finetuned", model_revision="v2.0.4")
        for cut_wav_path in wave_file_list:
            rec_result = inference_pipeline(cut_wav_path, output_dir="./output", granularity="utterance", extract_embedding=False)
            emotion_list.append(rec_result)
            scores = rec_result[0]["scores"]
            for key, value in zip(emotion_labels, scores):
                emotion_per_second[key].append(value)

        # Transcription
        trans_model = load_model()
        audio = audio_from_path(wav_path)
        ret = transcribe(trans_model, audio)

        self.write_wav_result(ret, emotion_list)

    def cut_wav(self, wav_path, duration):
        # Implementation of cut_wav method
        pass

    def write_wav_result(self, ret, emotion_list):
        # Implementation of write_wav_result method
        pass

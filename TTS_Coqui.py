from TTS.api import TTS

# Running a multi-speaker and multi-lingual model

# List available 🐸TTS models and choose the first one
model_name = TTS.list_models()[0]
# Init TTS
tts = TTS(model_name)


def test_TTS(count, example_text=''):
    tts.tts_to_file(text=example_text, speaker=tts.speakers[0], language=tts.languages[0],
                    file_path="output" + str(count) + ".wav")

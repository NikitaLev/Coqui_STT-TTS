#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import shlex
import subprocess
import sys
import wave
from timeit import default_timer as timer

import numpy as np
import sox
from stt import Model, version

try:
    from shlex import quote
except ImportError:
    from pipes import quote

wav_f = 'out.wav'

str_ex = 'the birch canoe slid on the smooth planks glue the sheet to the dark blue background it\'s easy to tell the depth of a well four hours of steady work faced us'



def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = "sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - ".format(
        quote(audio_path), desired_sample_rate
    )
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("SoX returned non-zero status: {}".format(e.stderr))
    except OSError as e:
        raise OSError(
            e.errno,
            "SoX not found, use {}hz files or install it: {}".format(
                desired_sample_rate, e.strerror
            ),
        )

    return desired_sample_rate, np.frombuffer(output, np.int16)


def _maybe_convert_wav(inp_filename, wav_filename):
    SAMPLE_RATE = 16000
    CHANNELS = 1
    if not os.path.exists(wav_filename):
        transformer = sox.Transformer()
        transformer.convert(samplerate=SAMPLE_RATE, n_channels=CHANNELS)
        try:
            transformer.build(inp_filename, wav_filename)
        except sox.core.SoxError:
            pass


def metadata_to_string(metadata):
    return "".join(token.text for token in metadata.tokens)


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def metadata_json_output(metadata):
    json_result = dict()
    json_result["transcripts"] = [
        {
            "confidence": transcript.confidence,
            "words": words_from_candidate_transcript(transcript),
        }
        for transcript in metadata.transcripts
    ]
    return json.dumps(json_result, indent=2)


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        print("Coqui STT ", version())
        exit(0)

#--model model_en.tflite --audio en_sample_1_16k.wav


def coquiSTT(audio, model):
    print("Loading model from file {}".format(model), file=sys.stderr)
    model_load_start = timer()
    # sphinx-doc: python_ref_model_start
    ds = Model(model)
    # sphinx-doc: python_ref_model_stop
    model_load_end = timer() - model_load_start
    print("Loaded model in {:.3}s.".format(model_load_end), file=sys.stderr)


    desired_sample_rate = ds.sampleRate()

    #_maybe_convert_wav(args.audio, wav_f)
    fin = wave.open(audio, "rb")
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print(
            "Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.".format(
                fs_orig, desired_sample_rate
            ),
            file=sys.stderr,
        )
        fs_new, audio = convert_samplerate(audio, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1 / fs_orig)
    fin.close()

    print("Running inference.", file=sys.stderr)
    inference_start = timer()

    res = ds.stt(audio)

    # sphinx-doc: python_ref_inference_stop
    inference_end = timer() - inference_start
    print(
        "Inference took %0.3fs for %0.3fs audio file." % (inference_end, audio_length),
        file=sys.stderr,
    )
    return res


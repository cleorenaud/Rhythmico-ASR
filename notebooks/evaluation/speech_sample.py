#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
"""
Speech recognition samples for the Microsoft Cognitive Services Speech SDK
"""

import json
import string
import time
import threading
import wave
import utils
import os
from dotenv import load_dotenv

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("""
    Importing the Speech SDK for Python failed.
    Refer to
    https://docs.microsoft.com/azure/cognitive-services/speech-service/quickstart-python for
    installation instructions.
    """)
    import sys
    sys.exit(1)


# Set up the subscription info for the Speech Service:
# Replace with your own subscription key and service region (e.g., "westus").
load_dotenv()  # Load environment variables from .env file

speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_REGION")


def speech_recognize_async_from_file(filename):
    """performs one-shot speech recognition asynchronously with input from an audio file"""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(filename=filename)
    # Creates a speech recognizer using a file as audio input.
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config, language='fr-FR')

    # Perform recognition. `recognize_async` does not block until recognition is complete,
    # so other tasks can be performed while recognition is running.
    # However, recognition stops when the first utterance has been recognized.
    # For long-running recognition, use continuous recognitions instead.
    result_future = speech_recognizer.recognize_once_async()

    print('recognition is running....')
    # Other tasks can be performed here...

    # Retrieve the recognition result. This blocks until recognition is complete.
    result = result_future.get()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


def speech_recognize_continuous_from_file(filename):
    """performs continuous speech recognition with input from an audio file"""
    # <SpeechContinuousRecognitionWithFile>
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(filename=filename)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # Stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()
    # </SpeechContinuousRecognitionWithFile>


def speech_recognize_continuous_async_from_microphone():
    """performs continuous speech recognition asynchronously with input from microphone"""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language='fr-FR')

    done = False

    def recognizing_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        print('RECOGNIZING: {}'.format(evt))

    def recognized_cb(evt: speechsdk.SpeechRecognitionEventArgs):
        print('RECOGNIZED: {}'.format(evt))

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(recognizing_cb)
    speech_recognizer.recognized.connect(recognized_cb)
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Perform recognition. `start_continuous_recognition_async asynchronously initiates continuous recognition operation,
    # Other tasks can be performed on this thread while recognition starts...
    # wait on result_future.get() to know when initialization is done.
    # Call stop_continuous_recognition_async() to stop recognition.
    result_future = speech_recognizer.start_continuous_recognition_async()

    result_future.get()  # wait for voidfuture, so we know engine initialization is done.
    print('Continuous Recognition is now running, say something.')

    while not done:
        # No real sample parallel work to do on this thread, so just wait for user to type stop.
        # Can't exit function or speech_recognizer will go out of scope and be destroyed while running.
        print('type "stop" then enter when done')
        stop = input()
        if (stop.lower() == "stop"):
            print('Stopping async recognition.')
            speech_recognizer.stop_continuous_recognition_async()
            break

    print("recognition stopped, main thread can exit now.")


def speech_recognition_with_pull_stream(filename):
    """gives an example how to use a pull audio stream to recognize speech from a custom audio
    source"""
    class WavFileReaderCallback(speechsdk.audio.PullAudioInputStreamCallback):
        """Example class that implements the Pull Audio Stream interface to recognize speech from
        an audio file"""
        def __init__(self, filename: str):
            super().__init__()
            self._file_h = wave.open(filename, mode=None)

            self.sample_width = self._file_h.getsampwidth()

            assert self._file_h.getnchannels() == 1
            assert self._file_h.getsampwidth() == 2
            assert self._file_h.getframerate() == 16000
            assert self._file_h.getcomptype() == 'NONE'

        def read(self, buffer: memoryview) -> int:
            """read callback function"""
            size = buffer.nbytes
            frames = self._file_h.readframes(size // self.sample_width)

            buffer[:len(frames)] = frames

            return len(frames)

        def close(self):
            """close callback function"""
            self._file_h.close()

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Specify the audio format
    wave_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16,
                                                    channels=1)

    # Setup the audio stream
    callback = WavFileReaderCallback(filename)
    stream = speechsdk.audio.PullAudioInputStream(callback, wave_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # Instantiate the speech recognizer with pull stream input
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # Stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()

    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()


def read_wave_header(file_path):
    with wave.open(file_path, 'rb') as audio_file:
        framerate = audio_file.getframerate()
        bits_per_sample = audio_file.getsampwidth() * 8
        num_channels = audio_file.getnchannels()
        return framerate, bits_per_sample, num_channels


def push_stream_writer(stream, filename):
    # The number of bytes to push per buffer
    n_bytes = 3200
    wav_fh = wave.open(filename)
    # Start pushing data until all data has been read from the file
    try:
        while True:
            frames = wav_fh.readframes(n_bytes // 2)
            print('read {} bytes'.format(len(frames)))
            if not frames:
                break
            stream.write(frames)
            time.sleep(.1)
    finally:
        wav_fh.close()
        stream.close()  # must be done to signal the end of stream


def speech_recognition_with_push_stream():
    """gives an example how to use a push audio stream to recognize speech from a custom audio
    source"""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Setup the audio stream
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # Instantiate the speech recognizer with push stream input
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    recognition_done = threading.Event()

    # Connect callbacks to the events fired by the speech recognizer
    def session_stopped_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('SESSION STOPPED: {}'.format(evt))
        recognition_done.set()

    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(session_stopped_cb)
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

    # Start push stream writer thread
    push_stream_writer_thread = threading.Thread(target=push_stream_writer, args=[stream])
    push_stream_writer_thread.start()

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()

    # Wait until all input processed
    recognition_done.wait()

    # Stop recognition and clean up
    speech_recognizer.stop_continuous_recognition()
    push_stream_writer_thread.join()


def speech_recognition_with_push_stream_mulaw(filename):
    """gives an example how to use a push mulaw audio stream to recognize speech from a custom audio
    source"""
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)

    # Setup the audio stream
    audio_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000,
                                                     bits_per_sample=8,
                                                     channels=1, 
                                                     wave_stream_format=speechsdk.AudioStreamWaveFormat.MULAW)
    stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    # Instantiate the speech recognizer with push stream input
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    recognition_done = threading.Event()

    # Connect callbacks to the events fired by the speech recognizer
    def session_stopped_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('SESSION STOPPED: {}'.format(evt))
        recognition_done.set()

    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(session_stopped_cb)
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    
    # Open the wav file and push it to the push stream.
    # NOTE the wav header must be skipped before pushing the data to the stream.
    with open(filename, 'rb') as audio_file:
        wav_header_size = utils.get_wav_header_size(filename)
        # Read the wave header
        header = audio_file.read(wav_header_size)
        # Read the audio data
        audio_data = audio_file.read()
        stream.write(audio_data)
        stream.close()

    # Wait until all input processed
    recognition_done.wait()

    # Stop recognition and clean up
    speech_recognizer.stop_continuous_recognition()


def pronunciation_assessment_continuous_from_file(filename):
    """Performs continuous pronunciation assessment asynchronously with input from an audio file.
        See more information at https://aka.ms/csspeech/pa"""

    import difflib
    import json

    # Creates an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region, language='fr-FR')
    audio_config = speechsdk.audio.AudioConfig(filename=filename)

    reference_text = "What's the weather like?"
    # Create pronunciation assessment config, set grading system, granularity and if enable miscue based on your requirement.
    enable_miscue = True
    enable_prosody_assessment = True
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=enable_miscue)
    if enable_prosody_assessment:
        pronunciation_config.enable_prosody_assessment()

    # Creates a speech recognizer using a file as audio input.
    language = 'fr-FR'
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language=language, audio_config=audio_config)
    # Apply pronunciation assessment config to speech recognizer
    pronunciation_config.apply_to(speech_recognizer)

    done = False
    recognized_words = []
    prosody_scores = []
    fluency_scores = []
    durations = []

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    def recognized(evt: speechsdk.SpeechRecognitionEventArgs):
        print("pronunciation assessment for: {}".format(evt.result.text))
        pronunciation_result = speechsdk.PronunciationAssessmentResult(evt.result)
        print("    Accuracy score: {}, prosody score: {}, pronunciation score: {}, completeness score : {}, fluency score: {}".format(
            pronunciation_result.accuracy_score, pronunciation_result.prosody_score, pronunciation_result.pronunciation_score,
            pronunciation_result.completeness_score, pronunciation_result.fluency_score
        ))
        nonlocal recognized_words, prosody_scores, fluency_scores, durations
        recognized_words += pronunciation_result.words
        fluency_scores.append(pronunciation_result.fluency_score)
        if pronunciation_result.prosody_score is not None:
            prosody_scores.append(pronunciation_result.prosody_score)
        json_result = evt.result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult)
        jo = json.loads(json_result)
        nb = jo["NBest"][0]
        durations.append(sum([int(w["Duration"]) for w in nb["Words"]]))

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognized.connect(recognized)
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # Stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous pronunciation assessment
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()

    reference_words = [w.strip(string.punctuation) for w in reference_text.lower().split()]

    # For continuous pronunciation assessment mode, the service won't return the words with `Insertion` or `Omission`
    # even if miscue is enabled.
    # We need to compare with the reference text after received all recognized words to get these error words.
    if enable_miscue:
        diff = difflib.SequenceMatcher(None, reference_words, [x.word.lower() for x in recognized_words])
        final_words = []
        for tag, i1, i2, j1, j2 in diff.get_opcodes():
            if tag in ['insert', 'replace']:
                for word in recognized_words[j1:j2]:
                    if word.error_type == 'None':
                        word._error_type = 'Insertion'
                    final_words.append(word)
            if tag in ['delete', 'replace']:
                for word_text in reference_words[i1:i2]:
                    word = speechsdk.PronunciationAssessmentWordResult({
                        'Word': word_text,
                        'PronunciationAssessment': {
                            'ErrorType': 'Omission',
                        }
                    })
                    final_words.append(word)
            if tag == 'equal':
                final_words += recognized_words[j1:j2]
    else:
        final_words = recognized_words

    # We can calculate whole accuracy by averaging
    final_accuracy_scores = []
    for word in final_words:
        if word.error_type == 'Insertion':
            continue
        else:
            final_accuracy_scores.append(word.accuracy_score)
    accuracy_score = sum(final_accuracy_scores) / len(final_accuracy_scores)
    # Re-calculate the prosody score by averaging
    if len(prosody_scores) == 0:
        prosody_score = float("nan")
    else:
        prosody_score = sum(prosody_scores) / len(prosody_scores)
    # Re-calculate fluency score
    fluency_score = sum([x * y for (x, y) in zip(fluency_scores, durations)]) / sum(durations)
    # Calculate whole completeness score
    completeness_score = len([w for w in recognized_words if w.error_type == "None"]) / len(reference_words) * 100
    completeness_score = completeness_score if completeness_score <= 100 else 100

    print('    Paragraph accuracy score: {}, prosody score: {}, completeness score: {}, fluency score: {}'.format(
        accuracy_score, prosody_score, completeness_score, fluency_score
    ))

    for idx, word in enumerate(final_words):
        print('    {}: word: {}\taccuracy score: {}\terror type: {};'.format(
            idx + 1, word.word, word.accuracy_score, word.error_type
        ))


def pronunciation_assessment_from_stream(filename):
    """Performs pronunciation assessment asynchronously with input from an audio stream.
        See more information at https://aka.ms/csspeech/pa"""

    # Creates an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region, language='fr-FR')

    # Setup the audio stream
    framerate, bits_per_sample, num_channels = read_wave_header(filename)
    format = speechsdk.audio.AudioStreamFormat(samples_per_second=framerate, bits_per_sample=bits_per_sample, channels=num_channels)
    stream = speechsdk.audio.PushAudioInputStream(format)
    audio_config = speechsdk.audio.AudioConfig(stream=stream)

    reference_text = "What's the weather like?"
    # Create pronunciation assessment config, set grading system, granularity and if enable miscue based on your requirement.
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True)
    pronunciation_config.enable_prosody_assessment()

    # Create a speech recognizer using a file as audio input.
    language = 'fr-FR'
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language=language, audio_config=audio_config)
    # Apply pronunciation assessment config to speech recognizer
    pronunciation_config.apply_to(speech_recognizer)

    # Start push stream writer thread
    push_stream_writer_thread = threading.Thread(target=push_stream_writer, args=[stream])
    push_stream_writer_thread.start()
    result = speech_recognizer.recognize_once_async().get()
    push_stream_writer_thread.join()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print('pronunciation assessment for: {}'.format(result.text))
        pronunciation_result = speechsdk.PronunciationAssessmentResult(result)
        print('    Accuracy score: {}, prosody score: {}, pronunciation score: {}, completeness score : {}, fluency score: {}'.format(
            pronunciation_result.accuracy_score, pronunciation_result.prosody_score, pronunciation_result.pronunciation_score,
            pronunciation_result.completeness_score, pronunciation_result.fluency_score
        ))
        print('  Word-level details:')
        for idx, word in enumerate(pronunciation_result.words):
            print('    {}: word: {}\taccuracy score: {}\terror type: {};'.format(
                idx + 1, word.word, word.accuracy_score, word.error_type
            ))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


def pronunciation_assessment_configured_with_json(filename):
    """Performs pronunciation assessment asynchronously with input from an audio file.
        See more information at https://aka.ms/csspeech/pa"""

    # Creates an instance of a speech config with specified subscription key and service region.
    # Note: The sample is for en-US language.
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioConfig(filename=filename)

    reference_text = "What's the weather like?"
    # Create pronunciation assessment config with json string (JSON format is not recommended)
    enable_miscue, enable_prosody = True, True
    config_json = {
        "GradingSystem": "HundredMark",
        "Granularity": "Phoneme",
        "Dimension": "Comprehensive",
        "ScenarioId": "",  # "" is the default scenario or ask product team for a customized one
        "EnableMiscue": enable_miscue,
        "EnableProsodyAssessment": enable_prosody,
        "NBestPhonemeCount": 0,  # > 0 to enable "spoken phoneme" mode, 0 to disable
    }
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(json_string=json.dumps(config_json))
    pronunciation_config.reference_text = reference_text

    # Create a speech recognizer using a file as audio input.
    language = 'fr-FR'
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, language=language, audio_config=audio_config)
    # Apply pronunciation assessment config to speech recognizer
    pronunciation_config.apply_to(speech_recognizer)

    result = speech_recognizer.recognize_once_async().get()

    # Check the result
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print('pronunciation assessment for: {}'.format(result.text))
        pronunciation_result = json.loads(result.properties.get(speechsdk.PropertyId.SpeechServiceResponse_JsonResult))
        print('assessment results:\n{}'.format(json.dumps(pronunciation_result, indent=4)))
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))


def pronunciation_assessment_with_content_assessment(filename):
    """Performs content assessment asynchronously with input from an audio file.
        See more information at https://aka.ms/csspeech/pa"""

    # Create an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    # Note: The sample is for en-US language.
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Generally, the waveform should longer than 20s and the content should be more than 3 sentences.
    audio_config = speechsdk.audio.AudioConfig(filename=filename)

    # Create pronunciation assessment config, set grading system, granularity and if enable miscue based on your requirement.
    topic = "the season of the fall"
    pronunciation_config = speechsdk.PronunciationAssessmentConfig(
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=False)
    pronunciation_config.enable_prosody_assessment()
    pronunciation_config.enable_content_assessment_with_topic(topic)

    # Create a speech recognizer using a file as audio input.
    language = 'fr-FR'
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, language=language, audio_config=audio_config)
    # Apply pronunciation assessment config to speech recognizer
    pronunciation_config.apply_to(speech_recognizer)

    done = False
    pron_results = []
    recognized_text = ""

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print("CLOSING on {}".format(evt))
        nonlocal done
        done = True

    def recognized(evt):
        nonlocal pron_results, recognized_text
        if (evt.result.reason == speechsdk.ResultReason.RecognizedSpeech or
                evt.result.reason == speechsdk.ResultReason.NoMatch):
            pron_results.append(speechsdk.PronunciationAssessmentResult(evt.result))
            if evt.result.text.strip().rstrip(".") != "":
                print(f"Recognizing: {evt.result.text}")
                recognized_text += " " + evt.result.text.strip()

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognized.connect(recognized)
    speech_recognizer.session_started.connect(lambda evt: print("SESSION STARTED: {}".format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print("SESSION STOPPED {}".format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print("CANCELED {}".format(evt)))
    # Stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous pronunciation assessment
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)
    speech_recognizer.stop_continuous_recognition()

    # Content assessment result is in the last pronunciation assessment block
    assert pron_results[-1].content_assessment_result is not None
    content_result = pron_results[-1].content_assessment_result
    print(f"Content Assessment for: {recognized_text.strip()}")
    print("Content Assessment results:\n"
          f"\tGrammar score: {content_result.grammar_score:.1f}\n"
          f"\tVocabulary score: {content_result.vocabulary_score:.1f}\n"
          f"\tTopic score: {content_result.topic_score:.1f}")
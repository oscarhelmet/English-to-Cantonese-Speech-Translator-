from flask import Flask, render_template, request, jsonify
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pygame  
import speech_recognition as sr
from google.cloud import texttospeech
import time 



app = Flask(__name__)

M2Ctokenizer = AutoTokenizer.from_pretrained("botisan-ai/mt5-translate-zh-yue")
M2Cmodel = AutoModelForSeq2SeqLM.from_pretrained("botisan-ai/mt5-translate-zh-yue")
tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')


model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base")


model = model.cuda()

LANG_TOKEN_MAPPING = {
    'en': '<en>',
    'zh': '<zh>'
}

special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load('final_model.pt', map_location=torch.device('cuda')))


def reg():
    recognizer = sr.Recognizer()


    with sr.Microphone() as source:
        print("Adjusting noise ")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Recording for 4 seconds")
        recorded_audio = recognizer.listen(source, timeout=4)
        print("Done recording")

    try:
        print("Recognizing the text")
        text = recognizer.recognize_google(
                recorded_audio, 
                language="en-US"
            )

        print("Decoded Text : {}".format(text))
        return text

    except Exception as ex:

        return ex


def M2C(mandarin_sentence):
    input_tokens = M2Ctokenizer.encode(mandarin_sentence, return_tensors="pt")

    translation_tokens = M2Cmodel.generate(input_tokens)

    cantonese_translation = M2Ctokenizer.decode(translation_tokens[0], skip_special_tokens=True)

    return cantonese_translation



def translate(sentence):

  
    def encode_input_str(text, target_lang, tokenizer, seq_len, lang_token_map=LANG_TOKEN_MAPPING):
        target_lang_token = lang_token_map[target_lang]
        input_ids = tokenizer.encode(
            text=target_lang_token + text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=seq_len
        )
        return input_ids[0]

  
    input_ids = encode_input_str(
        text=sentence,
        target_lang='zh',
        tokenizer=tokenizer,
        seq_len=model.config.max_length,
        lang_token_map=LANG_TOKEN_MAPPING
    )
    input_ids = input_ids.unsqueeze(0).cuda()

    output_tokens = model.generate(
        input_ids,
        num_beams=10,
        # num_return_sequences=1,
        # length_penalty=1,
        no_repeat_ngram_size=3
    )
    translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return translated_text


def tts(text):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'keyfile.json'

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="yue-HK",
        name="yue-HK-Standard-A",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    where = time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp3"


    with open(where, "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(where)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy(): 
        pygame.time.Clock().tick(10)

    return where





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    if request.method == 'POST':
        regText = reg()
        
        text = translate(regText)

        fianl_text = M2C(text)
        
        audio_path = tts(fianl_text)
        
        return jsonify({
            'final': fianl_text,
            'audio_path': audio_path})

if __name__ == '__main__':
    app.run(debug=True)
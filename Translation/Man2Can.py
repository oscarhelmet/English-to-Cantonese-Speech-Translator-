from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

M2Ctokenizer = AutoTokenizer.from_pretrained("botisan-ai/mt5-translate-zh-yue")
M2Cmodel = AutoModelForSeq2SeqLM.from_pretrained("botisan-ai/mt5-translate-zh-yue")



def M2C(mandarin_sentence):
    input_tokens = M2Ctokenizer.encode(mandarin_sentence, return_tensors="pt")

    translation_tokens = M2Cmodel.generate(input_tokens)

    cantonese_translation = M2Ctokenizer.decode(translation_tokens[0], skip_special_tokens=True)

    return cantonese_translation
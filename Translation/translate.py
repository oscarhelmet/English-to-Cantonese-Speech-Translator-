import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


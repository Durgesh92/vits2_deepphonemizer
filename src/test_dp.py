from dp.phonemizer import Phonemizer
import torch

from dp.preprocessing.text import SequenceTokenizer

text = "test one two three"


data = torch.load("en_us_cmudict_ipa_forward.pt",map_location=torch.device('cpu'))

config = data["config"]
phoneme_symbols = config["preprocessing"]["phoneme_symbols"]
keys = list(data.keys())

phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa_forward.pt')
phonemes = phonemizer(text, lang='en_us')

tokenizer = SequenceTokenizer(symbols=phoneme_symbols, languages=['en'], char_repeats=1,
                                    lowercase=True, append_start_end=True, end_token='<end>')

data = []
for x in phonemes:
    data.append(x)

print("data : ",data)
tokens = tokenizer(data, language='en')
print("tokens : ",tokens)
decoded = tokenizer.decode(tokens)

print("decoded : ",decoded)
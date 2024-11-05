import torch
import torch.nn as nn
from dp.preprocessing.text import Preprocessor
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from dp.model.utils import _make_len_mask, PositionalEncoding


class MihupPostProcessing(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:         # shape: [T, N]
        x = x.transpose(0, 1)
        y = x.argmax(2)
        y = y[y != 0]
        #y = torch.unique_consecutive(x) # This part was not supported by onnx. Have to write own logic on model output to mimic same
        return y


class ForwardTransformerHelper(nn.Module):
    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 d_model=512,
                 d_fft=1024,
                 layers=4,
                 dropout=0.1,
                 heads=1) -> None:
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=heads,
                                                dim_feedforward=d_fft,
                                                dropout=dropout,
                                                activation='relu')
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer,
                                          num_layers=layers,
                                          norm=encoder_norm)

        self.fc_out = nn.Linear(d_model, decoder_vocab_size)
        self.custom_mihup_postprocessing = MihupPostProcessing()

    def forward(self, input_tensor: torch.tensor):
        print()
        x = input_tensor.transpose(0, 1)  # shape: [T, N]
        src_pad_mask = _make_len_mask(x).to(x.device)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc_out(x)
        x = self.custom_mihup_postprocessing(x)
        return x

    @classmethod
    def from_config(cls, config: dict) -> 'ForwardTransformerHelper':
        preprocessor = Preprocessor.from_config(config)
        return ForwardTransformerHelper(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )

_checkpoint_path = "en_us_cmudict_ipa_forward.pt"
_device = torch.device('cpu')
checkpoint = torch.load(_checkpoint_path, map_location=_device)
model = ForwardTransformerHelper.from_config(config=checkpoint['config'])
model.load_state_dict(checkpoint['model'])
model.eval()

model_step = checkpoint['step']
print('Initializing phonemizer with model step {0}'.format(model_step))

input_dummy = torch.tensor([[1, 20, 20, 20, 23, 23, 23, 16, 16, 16, 16, 16, 16, 11, 11, 11, 16, 16, 16,  9,  9,  9,  2]])
# Torch inference
torch_output = model.forward(input_dummy)
print("torch_output: ", torch_output)

torch.onnx.export(model,               # model being run
                  input_dummy,                         # model input (or a tuple for multiple inputs)
                  "en_us_cmudict_ipa_forward.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=17,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['embedding'],
                  output_names=['custom_mihup_postprocessing'],
                  dynamic_axes={'embedding': {1: 'input_length'}, 'custom_mihup_postprocessing': {0:  'output_length'}})

import onnxmltools
import onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16
onnx_model = onnxmltools.utils.load_model("en_us_cmudict_ipa_forward.onnx")
onnx_model = convert_float_to_float16(onnx_model)
onnx.save(onnx_model,"en_us_cmudict_ipa_forward.onnx")
print("fp16 quant complete")

print("Model converted successfully")

# Inference example
import onnxruntime
x = torch.tensor([[ 1,  4,  4,  4, 10, 10, 10,  3,  3,  3, 21, 21, 21, 25, 25, 25,  3,  3,
          3, 22, 22, 22, 11, 11, 11,  2]])
onnx_runtime_input = x.detach().numpy()
ort_session = onnxruntime.InferenceSession("en_us_cmudict_ipa_forward.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: onnx_runtime_input}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)
import onnx
model = onnx.load("D:/vits2_deepphonemizer/models/en_us_cmudict_ipa_forward.onnx")
onnx.checker.check_model(model)
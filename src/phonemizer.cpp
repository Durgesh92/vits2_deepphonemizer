#include "dhwani.hpp"
#ifndef _Frees_ptr_opt_
#define _Frees_ptr_opt_
#endif
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>
#include <locale>
#include <codecvt>

const std::array<const char *, 1> input_names = {"text"};
const std::array<const char *, 1> output_names = {"output"};

// Define symbol-to-ID mapping using wide characters (Unicode)
std::unordered_map<wchar_t, int64_t> _symbol_to_id = {
    {L'_', 0}, {L';', 1}, {L':', 2}, {L',', 3}, {L'.', 4}, {L'!', 5}, {L'?', 6}, {L'¡', 7}, {L'¿', 8}, {L'—', 9},
    {L'…', 10}, {L'"', 11}, {L'«', 12}, {L'»', 13}, {L'“', 14}, {L'”', 15}, {L' ', 16}, {L'A', 17}, {L'B', 18},
    {L'C', 19}, {L'D', 20}, {L'E', 21}, {L'F', 22}, {L'G', 23}, {L'H', 24}, {L'I', 25}, {L'J', 26}, {L'K', 27},
    {L'L', 28}, {L'M', 29}, {L'N', 30}, {L'O', 31}, {L'P', 32}, {L'Q', 33}, {L'R', 34}, {L'S', 35}, {L'T', 36},
    {L'U', 37}, {L'V', 38}, {L'W', 39}, {L'X', 40}, {L'Y', 41}, {L'Z', 42}, {L'a', 43}, {L'b', 44}, {L'c', 45},
    {L'd', 46}, {L'e', 47}, {L'f', 48}, {L'g', 49}, {L'h', 50}, {L'i', 51}, {L'j', 52}, {L'k', 53}, {L'l', 54},
    {L'm', 55}, {L'n', 56}, {L'o', 57}, {L'p', 58}, {L'q', 59}, {L'r', 60}, {L's', 61}, {L't', 62}, {L'u', 63},
    {L'v', 64}, {L'w', 65}, {L'x', 66}, {L'y', 67}, {L'z', 68}, {L'ɑ', 69}, {L'ɐ', 70}, {L'ɒ', 71}, {L'æ', 72},
    {L'ɓ', 73}, {L'ʙ', 74}, {L'β', 75}, {L'ɔ', 76}, {L'ɕ', 77}, {L'ç', 78}, {L'ɗ', 79}, {L'ɖ', 80}, {L'ð', 81},
    {L'ʤ', 82}, {L'ə', 83}, {L'ɘ', 84}, {L'ɚ', 85}, {L'ɛ', 86}, {L'ɜ', 87}, {L'ɝ', 88}, {L'ɞ', 89}, {L'ɟ', 90},
    {L'ʄ', 91}, {L'ɡ', 92}, {L'ɠ', 93}, {L'ɢ', 94}, {L'ʛ', 95}, {L'ɦ', 96}, {L'ɧ', 97}, {L'ħ', 98}, {L'ɥ', 99},
    {L'ʜ', 100}, {L'ɨ', 101}, {L'ɪ', 102}, {L'ʝ', 103}, {L'ɭ', 104}, {L'ɬ', 105}, {L'ɫ', 106}, {L'ɮ', 107},
    {L'ʟ', 108}, {L'ɱ', 109}, {L'ɯ', 110}, {L'ɰ', 111}, {L'ŋ', 112}, {L'ɳ', 113}, {L'ɲ', 114}, {L'ɴ', 115},
    {L'ø', 116}, {L'ɵ', 117}, {L'ɸ', 118}, {L'θ', 119}, {L'œ', 120}, {L'ɶ', 121}, {L'ʘ', 122}, {L'ɹ', 123},
    {L'ɺ', 124}, {L'ɾ', 125}, {L'ɻ', 126}, {L'ʀ', 127}, {L'ʁ', 128}, {L'ɽ', 129}, {L'ʂ', 130}, {L'ʃ', 131},
    {L'ʈ', 132}, {L'ʧ', 133}, {L'ʉ', 134}, {L'ʊ', 135}, {L'ʋ', 136}, {L'ⱱ', 137}, {L'ʌ', 138}, {L'ɣ', 139},
    {L'ɤ', 140}, {L'ʍ', 141}, {L'χ', 142}, {L'ʎ', 143}, {L'ʏ', 144}, {L'ʑ', 145}, {L'ʐ', 146}, {L'ʒ', 147},
    {L'ʔ', 148}, {L'ʡ', 149}, {L'ʕ', 150}, {L'ʢ', 151}, {L'ǀ', 152}, {L'ǁ', 153}, {L'ǂ', 154}, {L'ǃ', 155},
    {L'ˈ', 156}, {L'ˌ', 157}, {L'ː', 158}, {L'ˑ', 159}, {L'ʼ', 160}, {L'ʴ', 161}, {L'ʰ', 162}, {L'ʱ', 163},
    {L'ʲ', 164}, {L'ʷ', 165}, {L'ˠ', 166}, {L'ˤ', 167}, {L'˞', 168}, {L'↓', 169}, {L'↑', 170}, {L'→', 171},
    {L'↗', 172}, {L'↘', 173}, {L'\'', 176}, {L'̩', 175}, {L'ᵻ', 177}
};


std::vector<float> softmax(const std::vector<float>& logits) {
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities(logits.size());

    float sum = 0.0f;
    for (float logit : logits) {
        sum += std::exp(logit - max_logit);
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit) / sum;
    }

    return probabilities;
}

std::unordered_map<std::string, std::vector<std::string>> process_dictionary(const std::string& dictonary_str) {
    std::unordered_map<std::string, std::vector<std::string>> dictionary;
    std::istringstream dictionary_stream(dictonary_str);

    std::string line;
    while (std::getline(dictionary_stream, line)) {
        std::stringstream line_stream(line);
        std::string word;
        line_stream >> word;
        std::vector<std::string> phonemes;
        std::string phoneme;
        while (line_stream >> phoneme) {
            phonemes.push_back(phoneme);
        }
        dictionary[word] = phonemes;
    }

    return dictionary;
}

namespace DeepPhonemizer {
    SequenceTokenizer::SequenceTokenizer(const std::vector<std::string>& symbols, const std::vector<std::string>& languages, int char_repeats, bool lowercase, bool append_start_end)
        : char_repeats(char_repeats), lowercase(lowercase), append_start_end(append_start_end), pad_token(" "), end_token("<end>") {
        
        tokens.push_back(pad_token);
        special_tokens.insert(pad_token);

        for (const auto& lang : languages) {
            std::string lang_token = "<" + lang + ">";
            tokens.push_back(lang_token);
            special_tokens.insert(lang_token);
        }

        tokens.push_back(end_token);
        end_index = tokens.size() - 1;

        for (const auto& symbol : symbols) {
            tokens.push_back(symbol);
        }
    }

    std::vector<int64_t> SequenceTokenizer::operator()(const std::string& sentence, const std::string& language) const {
        std::string processed_sentence = sentence;
        if (lowercase) {
            std::transform(processed_sentence.begin(), processed_sentence.end(), processed_sentence.begin(), ::tolower);
        }

        std::vector<int64_t> sequence;
        for (char c : processed_sentence) {
            std::string symbol(1, c);
            auto index = get_token(symbol);
            if (index != -1) {
                for (int i = 0; i < char_repeats; ++i) {
                    sequence.push_back(index);
                }
            }
        }

        if (append_start_end) {
            auto index = get_token("<" + language + ">");
            sequence.insert(sequence.begin(), index);
            sequence.push_back(end_index);
        }

        // Pad the sequence to the maximum length (50)
        int max_length = 50;
        while (sequence.size() < max_length) {
            sequence.push_back(pad_index);
        }

        if (sequence.size() > max_length) {
            sequence.resize(max_length);
        }

        return sequence;
    }

    std::vector<std::string> SequenceTokenizer::decode(const std::vector<int64_t>& sequence) const {
        std::vector<int64_t> processed_sequence;
        if (append_start_end) {
            processed_sequence.push_back(sequence.front());
            for (size_t i = 1; i < sequence.size() - 1; i += char_repeats) {
                processed_sequence.push_back(sequence[i]);
            }
            processed_sequence.push_back(sequence.back());
        } else {
            for (size_t i = 0; i < sequence.size(); i += char_repeats) {
                processed_sequence.push_back(sequence[i]);
            }
        }

        std::vector<std::string> decoded;
        for (int64_t token : processed_sequence) {
            if (token == end_index) {
                break;
            }
            decoded.push_back(tokens[token]);
        }

        return decoded;
    }

    std::vector<int64_t> SequenceTokenizer::clean(const std::vector<int64_t>& sequence) const {
        std::vector<int64_t> processed_sequence = sequence;

        // remove all special tokens from the sequence
        for (auto token : special_tokens) {
            auto special_token_index = get_token(token);
            if (special_token_index != -1) {
                processed_sequence.erase(std::remove(processed_sequence.begin(), processed_sequence.end(), special_token_index), processed_sequence.end());
            }
        }
        
        // extract everything between the start and end tokens
        auto end = std::find(processed_sequence.begin(), processed_sequence.end(), end_index);
        if (end != processed_sequence.end()) {
            processed_sequence.erase(end, processed_sequence.end());
        }

        // Remove consecutive duplicate tokens
        auto last = std::unique(processed_sequence.begin(), processed_sequence.end());
        processed_sequence.erase(last, processed_sequence.end());
        
        return processed_sequence;
    }

    int64_t SequenceTokenizer::get_token(const std::string& token) const {
        auto it = std::find(tokens.begin(), tokens.end(), token);

        if (it != tokens.end()) {
            return std::distance(tokens.begin(), it);
        }

        return -1;
    }

    Session::Session(const std::string& model_path, const std::string language, const bool use_dictionaries, const bool use_punctuation) {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DeepPhonemizer");
        env.DisableTelemetryEvents();

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        std::cout << "loading : " << model_path << std::endl;


        try {
            #ifdef _WIN32
                std::wstring dp_model_path_w(model_path.begin(), model_path.end());
                this->session = new Ort::Session(env, dp_model_path_w.c_str(), session_options);
            #else
                this->session = new Ort::Session(env, (const ORTCHAR_T*)model_path.c_str(), session_options);
            #endif
        }
        catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
            // Optionally, handle the exception further, e.g., clean up resources or rethrow
            return;  // Adjust as necessary for your application flow
        }

        //this->session = new Ort::Session(env, (const ORTCHAR_T *) model_path.c_str(), session_options);

        // Load metadata from the model
        Ort::ModelMetadata model_metadata = session->GetModelMetadata();
        Ort::AllocatorWithDefaultOptions allocator;

        std::string langs_str = model_metadata.LookupCustomMetadataMapAllocated("languages", allocator).get();

        std::vector<std::string> languages;
        std::stringstream languages_stream(langs_str);
        std::string language_buffer;
        while (languages_stream >> language_buffer) {
            languages.push_back(language_buffer);
        }

        std::string text_symbols_str = model_metadata.LookupCustomMetadataMapAllocated("text_symbols", allocator).get();

        std::vector<std::string> text_symbols;
        std::stringstream text_symbols_stream(text_symbols_str);
        std::string text_symbol_buffer;
        while (text_symbols_stream >> text_symbol_buffer) {
            text_symbols.push_back(text_symbol_buffer);
        }

        std::string phoneme_symbols_str = model_metadata.LookupCustomMetadataMapAllocated("phoneme_symbols", allocator).get();

        std::vector<std::string> phoneme_symbols;
        std::stringstream phoneme_symbols_stream(phoneme_symbols_str);
        std::string phoneme_symbol_buffer;
        while (phoneme_symbols_stream >> phoneme_symbol_buffer) {
            phoneme_symbols.push_back(phoneme_symbol_buffer);
        }

        if (use_dictionaries) {
            for (const auto& lang : languages) {
                std::string key = lang + "_dictionary";
                std::string dictonary_str = model_metadata.LookupCustomMetadataMapAllocated(key.c_str(), allocator).get();
                this->dictionaries[lang] = process_dictionary(dictonary_str);
            }
        }

        int char_repeats = model_metadata.LookupCustomMetadataMapAllocated("char_repeats", allocator).get()[0] - '0';

        bool lowercase = model_metadata.LookupCustomMetadataMapAllocated("lowercase", allocator).get()[0] == '1';

        if (std::find(languages.begin(), languages.end(), language) == languages.end()) {
            throw std::runtime_error("Language not supported.");
        }

        this->language = language;
        this->use_dictionaries = use_dictionaries;
        this->use_punctuation = use_punctuation;
        this->text_tokenizer = new SequenceTokenizer(text_symbols, languages, char_repeats, lowercase);
        this->phoneme_tokenizer = new SequenceTokenizer(phoneme_symbols, languages, 1, false);
    }

    Session::~Session() {
        delete session;
        delete text_tokenizer;
        delete phoneme_tokenizer;
    }

    std::vector<std::string> Session::g2p(const std::string& text) {
        // Convert input text to phonemes
        std::cout<<"g2p text : "<<text<<std::endl;
        std::vector<int64_t> phoneme_tokens = g2p_tokens(text);
        std::cout<<"got phoneme_tokens"<<std::endl;
        // Decode the phoneme tokens
        return phoneme_tokenizer->decode(phoneme_tokens);
    }

    // Convert text to sequence of IDs
    std::vector<int64_t> Session::text_to_sequence(const std::string &text) {
        // std::cout<<"text : "<<text<<std::endl;
        // std::vector<int64_t> sequence;
        // for (wchar_t symbol : text) {
        //     if (_symbol_to_id.find(symbol) != _symbol_to_id.end()) {
        //         sequence.push_back(_symbol_to_id[symbol]);
        //     }
        // }
        // return sequence;
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::wstring wtext = converter.from_bytes(text);
        // wtext = 
        std::wcout << L"Converted wstring: " << wtext << std::endl;
        std::vector<int64_t> mapped_sequence;

        // Iterate through each character in the string
        for (wchar_t ch : wtext) {
            if (_symbol_to_id.find(ch) != _symbol_to_id.end()) {
                mapped_sequence.push_back(_symbol_to_id[ch]);
            } else {
                // Handle unmapped characters, e.g., ignore them or add a placeholder like -1
                mapped_sequence.push_back(-1);  // -1 for unmapped characters
            }
        }

        // Print the mapped sequence
        std::wcout << L"Mapped sequence: ";
        for (int64_t id : mapped_sequence) {
            std::wcout << id << L" ";
        }
        std::wcout << std::endl;
        return mapped_sequence;
    }

    std::vector<int64_t> Session::g2p_tokens(const std::string& text) {
        // Clean the input text
        std::vector<std::string> words = clean_text(text);

        // Convert each word to phonemes
        std::vector<int64_t> phoneme_ids;
        for (const auto& word : words) {
            std::cout << "word: " << word << std::endl;
            std::vector<int64_t> word_phoneme_ids = g2p_tokens_internal(word);
            std::vector<int64_t> cleaned_word_phoneme_ids = phoneme_tokenizer->clean(word_phoneme_ids);
            phoneme_ids.insert(phoneme_ids.end(), cleaned_word_phoneme_ids.begin(), cleaned_word_phoneme_ids.end());
            if (use_punctuation) {
                auto back_token = phoneme_tokenizer->get_token(std::string(1, word.back()));

                // Check if the word ends with punctuation
                if (std::ispunct(word.back()) && back_token != -1) {
                    phoneme_ids.push_back(back_token);
                }
            }
            phoneme_ids.push_back(0);
        }

        return phoneme_ids;
    }

    std::vector<int64_t> Session::g2p_tokens_internal(const std::string& text) {
        // Check if the input text is longer than one character
        std::string key_text = text;
        std::transform(key_text.begin(), key_text.end(), key_text.begin(), ::tolower);
        key_text.erase(std::remove_if(key_text.begin(), key_text.end(), ::ispunct), key_text.end());
        // First check if word is in the dictionary
        if (dictionaries[language].count(key_text) && use_dictionaries) {
            auto token_str = dictionaries[language].at(key_text);

            std::vector<int64_t> tokens;
            for (const auto& token : token_str) {
                tokens.push_back(phoneme_tokenizer->get_token(token));
            }

            return tokens;
        }
        // Convert input text to tensor
        std::vector<Ort::Value> input_tensors;
        std::vector<int64_t> input_ids = text_tokenizer->operator()(text, language);
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // Create input tensor
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, 
            input_ids.data(), 
            input_ids.size(), 
            input_shape.data(), 
            input_shape.size()
        ));
        // Run the model
        std::vector<Ort::Value> output_tensors = session->Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            input_tensors.data(), 
            input_names.size(), 
            output_names.data(), 
            output_names.size()
        );
        // Check if output tensor is valid
        if (output_tensors.empty()) {
            throw std::runtime_error("No output tensor returned from the model.");
        }
        // Process the output tensor
        const float* output_data = output_tensors.front().GetTensorData<float>();
        std::vector<int64_t> output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        // Ensure the output shape is as expected: {1, 50, 53}
        if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] != 50 || output_shape[2] != 53) {
            throw std::runtime_error("Unexpected output shape from the model.");
        }
        // Decode the output: find the index with the highest probability at each position
        std::vector<int64_t> output_ids_vector(output_shape[1]);
        for (size_t i = 0; i < output_shape[1]; ++i) {
            std::vector<float> logits(output_data + i * output_shape[2], output_data + (i + 1) * output_shape[2]);
            std::vector<float> probabilities = softmax(logits);

            auto max_prob_iter = std::max_element(probabilities.begin(), probabilities.end());
            output_ids_vector[i] = std::distance(probabilities.begin(), max_prob_iter);
        }

        return output_ids_vector;
    }
}
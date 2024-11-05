#ifndef DHWANI_H
#define DHWANI_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef WIN32
   #define DHWANI_EXPORT __declspec(dllexport)
#else
   #define DHWANI_EXPORT __attribute__((visibility("default"))) __attribute__((used))
#endif

typedef struct {
   const char* language;
   const unsigned char use_dictionaries;
   const unsigned char use_punctuation;
} dhwani_g2p_options_t;

DHWANI_EXPORT int dhwani_g2p_init(const char* model_path, dhwani_g2p_options_t options);

DHWANI_EXPORT char* dhwani_g2p(const char* text);

DHWANI_EXPORT int* dhwani_g2p_tokens(const char* text);

DHWANI_EXPORT void dhwani_g2p_free(void);

DHWANI_EXPORT int dhwani_tts_init(const char* model_path);

DHWANI_EXPORT void dhwani_tts(const char* text, const char* output_path);

DHWANI_EXPORT void dhwani_tts_free(void);

#ifdef __cplusplus
}
#endif

#endif // DHWANI_H
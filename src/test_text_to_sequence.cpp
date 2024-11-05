#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <locale>
#include <codecvt>

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

int main() {
    // Wide string with Unicode characters
    std::wstring text = L"həloʊ wɝːld, haʊ ɑːr juː? naʊ tɛkst ɛntri kənteɪnz ðə tɛkst ɛntɝd baɪ ðə juːzɝ.";
    
    // Vector to store the mapped numbers
    std::vector<int64_t> mapped_sequence;

    // Iterate through each character in the string
    for (wchar_t ch : text) {
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

    return 0;
}

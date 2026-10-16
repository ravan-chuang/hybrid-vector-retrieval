#include "retrieval/corpus_loader.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace retrieval {

namespace {

uint32_t read_u32(std::ifstream& file) {
    unsigned char bytes[4];

    if (!file.read(reinterpret_cast<char*>(bytes), 4)) {
        throw std::runtime_error("Unexpected end of file");
    }

    return static_cast<uint32_t>(bytes[0])
         | (static_cast<uint32_t>(bytes[1]) << 8)
         | (static_cast<uint32_t>(bytes[2]) << 16)
         | (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t read_u64(std::ifstream& file) {
    const uint64_t low = read_u32(file);
    const uint64_t high = read_u32(file);

    return low | (high << 32);
}

}

std::size_t load_corpus(
    const std::string& path,
    SearchEngine& engine
) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Cannot open corpus: " + path);
    }

    char magic[4];

    if (!file.read(magic, 4) ||
        std::string(magic, 4) != "MSRC") {
        throw std::runtime_error("Invalid corpus magic");
    }

    const uint32_t version = read_u32(file);

    if (version != 1) {
        throw std::runtime_error("Unsupported corpus version");
    }

    const uint64_t count = read_u64(file);

    if (count > 100000000ULL) {
        throw std::runtime_error("Unreasonable document count");
    }

    for (uint64_t i = 0; i < count; ++i) {
        const uint32_t doc_id = read_u32(file);
        const uint32_t length = read_u32(file);

        if (length > 16 * 1024 * 1024) {
            throw std::runtime_error("Document exceeds size limit");
        }

        std::string content(length, '\0');

        if (!file.read(content.data(), length)) {
            throw std::runtime_error("Truncated document");
        }

        engine.add_document(doc_id, content);
    }

    if (file.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Unexpected trailing data");
    }

    return static_cast<std::size_t>(count);
}

}

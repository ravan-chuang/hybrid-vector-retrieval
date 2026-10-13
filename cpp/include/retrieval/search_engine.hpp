#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace retrieval {

struct Posting {
    uint32_t doc_id;
    uint32_t term_frequency;
};

struct SearchResult {
    uint32_t doc_id;
    double score;
};

class SearchEngine {
public:
    void add_document(uint32_t doc_id, const std::string& text);

    std::vector<SearchResult> search(
        const std::string& query,
        std::size_t top_k
    ) const;

    std::size_t document_count() const;
    std::size_t vocabulary_size() const;
    std::size_t posting_count() const;
    std::size_t document_frequency(const std::string& term) const;

private:
    static std::vector<std::string> tokenize(
        const std::string& text
    );

    std::unordered_map<
        std::string,
        std::vector<Posting>
    > inverted_index_;

    std::unordered_map<uint32_t, std::size_t> document_lengths_;

    std::size_t total_document_length_ = 0;

    double k1_ = 1.2;
    double b_ = 0.75;
};

}  // namespace retrieval

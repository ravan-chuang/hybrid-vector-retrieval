#include "retrieval/search_engine.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace retrieval {

std::vector<std::string> SearchEngine::tokenize(
    const std::string& text
) {
    std::vector<std::string> tokens;
    std::string current;

    for (unsigned char ch : text) {
        if (std::isalnum(ch)) {
            current.push_back(
                static_cast<char>(std::tolower(ch))
            );
        } else if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}

void SearchEngine::add_document(
    uint32_t doc_id,
    const std::string& text
) {
    if (document_lengths_.contains(doc_id)) {
        throw std::invalid_argument("Duplicate document ID");
    }

    if (internal_to_external_.size() >= UINT32_MAX) {
        throw std::overflow_error("Too many documents");
    }

    auto tokens = tokenize(text);

    std::unordered_map<std::string, uint32_t> frequencies;

    for (const auto& token : tokens) {
        ++frequencies[token];
    }

    const uint32_t internal_id =
        static_cast<uint32_t>(internal_to_external_.size());

    external_to_internal_[doc_id] = internal_id;
    internal_to_external_.push_back(doc_id);
    internal_document_lengths_.push_back(tokens.size());

    for (const auto& [term, frequency] : frequencies) {
        inverted_index_[term].push_back({
            doc_id,
            frequency,
            internal_id
        });
    }

    document_lengths_[doc_id] = tokens.size();
    total_document_length_ += tokens.size();
}

std::vector<SearchResult> SearchEngine::search(
    const std::string& query,
    std::size_t top_k
) const {
    if (top_k == 0 || document_lengths_.empty()) {
        return {};
    }

    auto query_tokens = tokenize(query);

    std::unordered_set<std::string> unique_terms(
        query_tokens.begin(),
        query_tokens.end()
    );

    std::unordered_map<uint32_t, double> scores;

    const double N =
        static_cast<double>(document_lengths_.size());

    const double avgdl =
        static_cast<double>(total_document_length_) / N;

    for (const auto& term : unique_terms) {
        auto it = inverted_index_.find(term);

        if (it == inverted_index_.end()) {
            continue;
        }

        const auto& postings = it->second;

        const double df =
            static_cast<double>(postings.size());

        const double idf = std::log(
            1.0 + (N - df + 0.5) / (df + 0.5)
        );

        for (const auto& posting : postings) {
            const double tf =
                static_cast<double>(posting.term_frequency);

            const double dl =
                static_cast<double>(
                    document_lengths_.at(posting.doc_id)
                );

            const double denominator =
                tf + k1_ * (
                    1.0 - b_ + b_ * dl / avgdl
                );

            const double score =
                idf * (tf * (k1_ + 1.0)) / denominator;

            scores[posting.doc_id] += score;
        }
    }

    auto better = [](const SearchResult& a, const SearchResult& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.doc_id < b.doc_id;
    };

    std::priority_queue<
        SearchResult,
        std::vector<SearchResult>,
        decltype(better)
    > heap(better);

    for (const auto& [doc_id, score] : scores) {
        SearchResult result{doc_id, score};

        if (heap.size() < top_k) {
            heap.push(result);
        } else if (better(result, heap.top())) {
            heap.pop();
            heap.push(result);
        }
    }

    std::vector<SearchResult> results;

    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }

    std::sort(
        results.begin(),
        results.end(),
        better
    );

    return results;
}


std::vector<SearchResult> SearchEngine::search_dense(
    const std::string& query,
    std::size_t top_k,
    SearchWorkspace& workspace
) const {
    if (top_k == 0 || document_lengths_.empty()) {
        return {};
    }

    const std::size_t N_docs = internal_to_external_.size();

    if (workspace.scores.size() < N_docs) {
        workspace.scores.resize(N_docs, 0.0);
        workspace.generations.resize(N_docs, 0);
    }

    if (workspace.current_generation == UINT32_MAX) {
        std::fill(
            workspace.generations.begin(),
            workspace.generations.end(),
            0
        );

        workspace.current_generation = 0;
    }

    ++workspace.current_generation;
    workspace.touched_docs.clear();

    auto query_tokens = tokenize(query);

    std::unordered_set<std::string> unique_terms(
        query_tokens.begin(),
        query_tokens.end()
    );

    const double N = static_cast<double>(N_docs);

    const double avgdl =
        static_cast<double>(total_document_length_) / N;

    for (const auto& term : unique_terms) {
        auto it = inverted_index_.find(term);

        if (it == inverted_index_.end()) {
            continue;
        }

        const auto& postings = it->second;

        const double df =
            static_cast<double>(postings.size());

        const double idf = std::log(
            1.0 + (N - df + 0.5) / (df + 0.5)
        );

        for (const auto& posting : postings) {
            const uint32_t internal_id = posting.internal_id;

            if (workspace.generations[internal_id] !=
                workspace.current_generation) {

                workspace.generations[internal_id] =
                    workspace.current_generation;

                workspace.scores[internal_id] = 0.0;

                workspace.touched_docs.push_back(internal_id);
            }

            const double tf =
                static_cast<double>(posting.term_frequency);

            const double dl =
                static_cast<double>(
                    internal_document_lengths_[internal_id]
                );

            const double denominator =
                tf + k1_ * (
                    1.0 - b_ + b_ * dl / avgdl
                );

            const double score =
                idf * (tf * (k1_ + 1.0)) / denominator;

            workspace.scores[internal_id] += score;
        }
    }

    auto better = [](const SearchResult& a, const SearchResult& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.doc_id < b.doc_id;
    };

    std::priority_queue<
        SearchResult,
        std::vector<SearchResult>,
        decltype(better)
    > heap(better);

    for (const uint32_t internal_id : workspace.touched_docs) {
        const SearchResult result{
            internal_to_external_[internal_id],
            workspace.scores[internal_id]
        };

        if (heap.size() < top_k) {
            heap.push(result);
        } else if (better(result, heap.top())) {
            heap.pop();
            heap.push(result);
        }
    }

    std::vector<SearchResult> results;

    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }

    std::sort(results.begin(), results.end(), better);

    return results;
}

std::size_t SearchEngine::document_count() const {
    return document_lengths_.size();
}

std::size_t SearchEngine::vocabulary_size() const {
    return inverted_index_.size();
}

std::size_t SearchEngine::posting_count() const {
    std::size_t total = 0;

    for (const auto& [term, postings] : inverted_index_) {
        total += postings.size();
    }

    return total;
}

std::size_t SearchEngine::document_frequency(
    const std::string& term
) const {
    const auto tokens = tokenize(term);

    if (tokens.size() != 1) {
        return 0;
    }

    const auto it = inverted_index_.find(tokens[0]);

    if (it == inverted_index_.end()) {
        return 0;
    }

    return it->second.size();
}

}  // namespace retrieval

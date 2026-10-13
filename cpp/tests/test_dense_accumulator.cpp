#include "retrieval/search_engine.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void compare_results(
    const std::vector<retrieval::SearchResult>& baseline,
    const std::vector<retrieval::SearchResult>& optimized
) {
    if (baseline.size() != optimized.size()) {
        throw std::runtime_error("Result count mismatch");
    }

    for (std::size_t i = 0; i < baseline.size(); ++i) {
        if (baseline[i].doc_id != optimized[i].doc_id) {
            throw std::runtime_error("Ranking mismatch");
        }

        if (std::abs(
            baseline[i].score - optimized[i].score
        ) > 1e-9) {
            throw std::runtime_error("BM25 score mismatch");
        }
    }
}

int main() {
    try {
        retrieval::SearchEngine engine;
        retrieval::SearchWorkspace workspace;

        engine.add_document(10, "machine learning retrieval");
        engine.add_document(200, "machine learning machine");
        engine.add_document(9999, "information retrieval systems");
        engine.add_document(50000, "database indexing retrieval");

        const std::vector<std::string> queries = {
            "machine learning",
            "retrieval",
            "database indexing",
            "unknown",
            "",
            "machine machine learning"
        };

        for (const auto& query : queries) {
            compare_results(
                engine.search(query, 10),
                engine.search_dense(query, 10, workspace)
            );
        }

        for (std::size_t k : {0, 1, 2, 3, 10}) {
            compare_results(
                engine.search("machine retrieval", k),
                engine.search_dense("machine retrieval", k, workspace)
            );
        }

        std::cout
            << "[PASS] Dense accumulator correctness\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}

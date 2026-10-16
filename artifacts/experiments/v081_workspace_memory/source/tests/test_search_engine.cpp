#include "retrieval/search_engine.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_basic_ranking() {
    retrieval::SearchEngine engine;

    engine.add_document(1, "natural disaster causes damage");
    engine.add_document(2, "disaster recovery and prevention");
    engine.add_document(3, "nature and environment");

    require(engine.document_count() == 3, "Document count mismatch");

    const auto results = engine.search("disaster recovery", 3);

    require(results.size() == 2, "Expected two matching documents");
    require(results[0].doc_id == 2, "Incorrect first result");
    require(results[1].doc_id == 1, "Incorrect second result");
    require(results[0].score > results[1].score, "Incorrect ranking");

    std::cout << "[PASS] Basic BM25 ranking\n";
}

void test_edge_cases() {
    retrieval::SearchEngine engine;

    engine.add_document(1, "machine learning");
    engine.add_document(2, "machine learning");
    engine.add_document(3, "");

    require(engine.document_count() == 3, "Empty document not indexed");

    require(engine.search("", 10).empty(), "Empty query should return nothing");
    require(engine.search("unknown", 10).empty(), "Unknown term should return nothing");
    require(engine.search("machine", 0).empty(), "Top-K zero should return nothing");

    const auto results = engine.search("machine", 100);

    require(results.size() == 2, "Top-K greater than matches failed");

    // Identical scores must use ascending document ID.
    require(results[0].doc_id == 1, "Tie-breaking failed");
    require(results[1].doc_id == 2, "Tie-breaking failed");

    bool duplicate_rejected = false;

    try {
        engine.add_document(1, "duplicate");
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }

    require(duplicate_rejected, "Duplicate ID was accepted");

    std::cout << "[PASS] Edge cases\n";
}

void test_top_k() {
    retrieval::SearchEngine engine;

    for (uint32_t i = 1; i <= 1000; ++i) {
        std::string text = "retrieval system";

        for (uint32_t j = 0; j < i % 10; ++j) {
            text += " retrieval";
        }

        engine.add_document(i, text);
    }

    const auto top10 = engine.search("retrieval", 10);
    const auto all = engine.search("retrieval", 1000);

    require(top10.size() == 10, "Top-10 size mismatch");
    require(all.size() == 1000, "Full ranking size mismatch");

    for (std::size_t i = 0; i < 10; ++i) {
        require(
            top10[i].doc_id == all[i].doc_id,
            "Top-K differs from full ranking"
        );

        require(
            std::abs(top10[i].score - all[i].score) < 1e-12,
            "Top-K score mismatch"
        );
    }

    std::cout << "[PASS] Top-K consistency\n";
}

int main() {
    try {
        test_basic_ranking();
        test_edge_cases();
        test_top_k();

        std::cout << "\nAll C++ Retrieval Core tests passed.\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] " << e.what() << '\n';
        return 1;
    }
}

#include "retrieval/search_engine.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

void require_close(
    double actual,
    double expected,
    double tolerance,
    const std::string& message
) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        retrieval::SearchEngine engine;

        engine.add_document(
            1,
            "natural disaster causes damage"
        );

        engine.add_document(
            2,
            "disaster recovery and prevention"
        );

        engine.add_document(
            3,
            "nature and environment"
        );

        const auto results = engine.search(
            "disaster recovery",
            3
        );

        if (results.size() != 2) {
            throw std::runtime_error(
                "Incorrect number of results"
            );
        }

        if (results[0].doc_id != 2 ||
            results[1].doc_id != 1) {
            throw std::runtime_error(
                "Incorrect document ranking"
            );
        }

        require_close(
            results[0].score,
            1.398810,
            1e-5,
            "Document 2 BM25 mismatch"
        );

        require_close(
            results[1].score,
            0.453151,
            1e-5,
            "Document 1 BM25 mismatch"
        );

        std::cout
            << "[PASS] BM25 reference scores\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr
            << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}

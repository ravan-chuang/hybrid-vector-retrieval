#include "retrieval/search_engine.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using retrieval::SearchEngine;
using retrieval::SearchResult;
using retrieval::SearchWorkspace;

struct TestCase {
    std::string query;
    std::size_t top_k;
    std::vector<SearchResult> expected;
};

bool equal_results(
    const std::vector<SearchResult>& a,
    const std::vector<SearchResult>& b
) {
    if (a.size() != b.size()) {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].doc_id != b[i].doc_id) {
            return false;
        }

        if (std::abs(a[i].score - b[i].score) > 1e-9) {
            return false;
        }
    }

    return true;
}

int main() {
    try {
        SearchEngine engine;

        constexpr uint32_t N_DOCS = 10000;

        for (uint32_t i = 1; i <= N_DOCS; ++i) {
            std::string text = "retrieval system";

            if (i % 2 == 0) {
                text += " machine learning";
            }

            if (i % 3 == 0) {
                text += " database indexing";
            }

            if (i % 5 == 0) {
                text += " neural search";
            }

            engine.add_document(i, text);
        }

        const std::vector<std::string> queries = {
            "retrieval",
            "machine learning",
            "database indexing",
            "neural search",
            "machine database",
            "unknown",
            ""
        };

        std::vector<TestCase> cases;

        for (const auto& query : queries) {
            for (std::size_t k : {0, 1, 5, 10, 50}) {
                cases.push_back({
                    query,
                    k,
                    engine.search(query, k)
                });
            }
        }

        constexpr int THREADS = 8;
        constexpr int ROUNDS = 100;

        std::atomic<int> failures{0};
        std::atomic<int> completed{0};

        std::vector<std::thread> workers;
        workers.reserve(THREADS);

        for (int t = 0; t < THREADS; ++t) {
            workers.emplace_back([&]() {
                SearchWorkspace workspace;

                for (int round = 0; round < ROUNDS; ++round) {
                    for (const auto& test : cases) {
                        const auto actual = engine.search_dense(
                            test.query,
                            test.top_k,
                            workspace
                        );

                        if (!equal_results(actual, test.expected)) {
                            ++failures;
                        }

                        ++completed;
                    }
                }
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }

        std::cout
            << "Documents: " << engine.document_count() << '\n'
            << "Threads: " << THREADS << '\n'
            << "Searches: " << completed.load() << '\n'
            << "Failures: " << failures.load() << '\n';

        if (failures.load() != 0) {
            std::cerr << "[FAIL] Parallel search parity\n";
            return 1;
        }

        std::cout << "[PASS] Parallel search parity\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}

#include "retrieval/corpus_loader.hpp"
#include "retrieval/search_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());

    const double position = p * (values.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, values.size() - 1);

    return values[lower] * (1.0 - (position - lower))
         + values[upper] * (position - lower);
}

void verify_results(
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

        if (std::abs(baseline[i].score - optimized[i].score) > 1e-8) {
            throw std::runtime_error("BM25 score mismatch");
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: benchmark_dense <corpus.bin>\n";
        return 1;
    }

    try {
        retrieval::SearchEngine engine;
        retrieval::SearchWorkspace workspace;

        std::cout << std::fixed << std::setprecision(3);

        std::cout << "Loading corpus...\n";

        const auto build_start = Clock::now();

        const auto count = retrieval::load_corpus(argv[1], engine);

        const auto build_end = Clock::now();

        const double build_seconds =
            std::chrono::duration<double>(
                build_end - build_start
            ).count();

        std::cout << "Documents: " << count << '\n';
        std::cout << "Build time: " << build_seconds << " s\n";

        const std::vector<std::string> queries = {
            "nature",
            "science",
            "history",
            "machine learning",
            "Manhattan Project",
            "heart disease",
            "database indexing"
        };

        constexpr int warmup = 10;
        constexpr int repeats = 200;

        std::cout << "\n=== Hash Map vs Dense Accumulator ===\n";

        for (const auto& query : queries) {
            const auto baseline_results = engine.search(query, 10);
            const auto dense_results =
                engine.search_dense(query, 10, workspace);

            verify_results(baseline_results, dense_results);

            for (int i = 0; i < warmup; ++i) {
                engine.search(query, 10);
                engine.search_dense(query, 10, workspace);
            }

            std::vector<double> baseline_ms;
            std::vector<double> dense_ms;

            baseline_ms.reserve(repeats);
            dense_ms.reserve(repeats);

            for (int i = 0; i < repeats; ++i) {
                // Alternate execution order to reduce systematic bias.
                const bool baseline_first = (i % 2 == 0);

                auto run_baseline = [&]() {
                    const auto start = Clock::now();
                    const auto results = engine.search(query, 10);
                    const auto end = Clock::now();

                    verify_results(baseline_results, results);

                    baseline_ms.push_back(
                        std::chrono::duration<double, std::milli>(
                            end - start
                        ).count()
                    );
                };

                auto run_dense = [&]() {
                    const auto start = Clock::now();
                    const auto results =
                        engine.search_dense(query, 10, workspace);
                    const auto end = Clock::now();

                    verify_results(baseline_results, results);

                    dense_ms.push_back(
                        std::chrono::duration<double, std::milli>(
                            end - start
                        ).count()
                    );
                };

                if (baseline_first) {
                    run_baseline();
                    run_dense();
                } else {
                    run_dense();
                    run_baseline();
                }
            }

            const double baseline_p50 =
                percentile(baseline_ms, 0.50);

            const double dense_p50 =
                percentile(dense_ms, 0.50);

            const double baseline_p95 =
                percentile(baseline_ms, 0.95);

            const double dense_p95 =
                percentile(dense_ms, 0.95);

            const double speedup = baseline_p50 / dense_p50;

            std::cout
                << std::left << std::setw(22) << query
                << " Hash P50=" << std::setw(9) << baseline_p50
                << " Dense P50=" << std::setw(9) << dense_p50
                << " Speedup=" << std::setw(7) << speedup
                << " Hash P95=" << std::setw(9) << baseline_p95
                << " Dense P95=" << dense_p95
                << '\n';
        }

        std::cout << "\nAll ranking checks passed.\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

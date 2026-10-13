#include "retrieval/corpus_loader.hpp"
#include "retrieval/search_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const double position = p * (values.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, values.size() - 1);

    const double fraction = position - lower;

    return values[lower] * (1.0 - fraction)
         + values[upper] * fraction;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: benchmark_index <corpus.bin>\n";
        return 1;
    }

    try {
        retrieval::SearchEngine engine;

        const auto build_start = Clock::now();

        const auto count = retrieval::load_corpus(
            argv[1],
            engine
        );

        const auto build_end = Clock::now();

        const double build_seconds =
            std::chrono::duration<double>(
                build_end - build_start
            ).count();

        std::cout << std::fixed << std::setprecision(3);

        std::cout << "\n=== Index Statistics ===\n";

        std::cout
            << "Documents: " << count << '\n'
            << "Vocabulary size: "
            << engine.vocabulary_size() << '\n'
            << "Posting count: "
            << engine.posting_count() << '\n'
            << "Build time: "
            << build_seconds << " s\n"
            << "Indexing throughput: "
            << count / build_seconds
            << " docs/s\n";

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

        std::cout << "\n=== Query Latency ===\n";

        for (const auto& query : queries) {
            for (int i = 0; i < warmup; ++i) {
                engine.search(query, 10);
            }

            std::vector<double> latencies;
            latencies.reserve(repeats);

            std::size_t result_count = 0;

            for (int i = 0; i < repeats; ++i) {
                const auto start = Clock::now();

                const auto results = engine.search(query, 10);

                const auto end = Clock::now();

                result_count = results.size();

                const double elapsed_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(end - start).count();

                latencies.push_back(elapsed_ms);
            }

            const double p50 = percentile(latencies, 0.50);
            const double p95 = percentile(latencies, 0.95);
            const double p99 = percentile(latencies, 0.99);

            std::cout
                << std::left << std::setw(22) << query
                << " P50=" << std::setw(9) << p50
                << " P95=" << std::setw(9) << p95
                << " P99=" << std::setw(9) << p99
                << " results=" << result_count
                << '\n';
        }

        std::cout << "\n=== Document Frequency ===\n";

        const std::vector<std::string> terms = {
            "nature",
            "science",
            "history",
            "machine",
            "learning",
            "heart",
            "disease",
            "database"
        };

        for (const auto& term : terms) {
            std::cout
                << std::left << std::setw(15) << term
                << " DF="
                << engine.document_frequency(term)
                << '\n';
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

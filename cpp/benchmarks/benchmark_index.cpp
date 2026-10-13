#include "retrieval/corpus_loader.hpp"
#include "retrieval/search_engine.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: benchmark_index <corpus.bin>\n";
        return 1;
    }

    try {
        retrieval::SearchEngine engine;

        const auto start = Clock::now();

        const auto count = retrieval::load_corpus(
            argv[1],
            engine
        );

        const auto end = Clock::now();

        const double seconds =
            std::chrono::duration<double>(end - start).count();

        std::cout << std::fixed << std::setprecision(3);

        std::cout << "Documents indexed: " << count << '\n';
        std::cout << "Index build time: " << seconds << " s\n";
        std::cout << "Documents / second: "
                  << count / seconds << '\n';

        const std::vector<std::string> queries = {
            "nature",
            "science",
            "history",
            "machine learning",
            "Manhattan Project",
            "heart disease",
            "database indexing"
        };

        std::cout << "\nSearch benchmark:\n";

        for (const auto& query : queries) {
            constexpr int repeats = 10;

            // Warmup
            engine.search(query, 10);

            double total_ms = 0.0;
            std::size_t result_count = 0;

            for (int i = 0; i < repeats; ++i) {
                const auto q_start = Clock::now();

                const auto results = engine.search(query, 10);

                const auto q_end = Clock::now();

                result_count = results.size();

                total_ms += std::chrono::duration<
                    double,
                    std::milli
                >(q_end - q_start).count();
            }

            std::cout
                << std::setw(22) << std::left << query
                << " avg=" << std::setw(10)
                << total_ms / repeats
                << " ms"
                << " results=" << result_count
                << '\n';
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

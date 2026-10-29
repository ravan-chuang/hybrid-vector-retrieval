#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "retrieval/search_engine.hpp"
#include "retrieval/corpus_loader.hpp"

namespace py = pybind11;

namespace {

class PythonSearchEngine {
public:
    std::size_t load_corpus(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (sealed_) {
            throw std::runtime_error(
                "Cannot load corpus after search initialization."
            );
        }

        if (engine_.document_count() != 0) {
            throw std::runtime_error(
                "Corpus loading requires an empty engine."
            );
        }

        return retrieval::load_corpus(path, engine_);
    }

    void add_document(
        uint32_t doc_id,
        const std::string& text
    ) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (sealed_) {
            throw std::runtime_error(
                "Cannot add documents after search initialization."
            );
        }

        engine_.add_document(doc_id, text);
    }

    std::vector<retrieval::SearchResult> search(
        const std::string& query,
        std::size_t top_k
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        sealed_ = true;

        return engine_.search(query, top_k);
    }

    std::vector<retrieval::SearchResult> search_dense(
        const std::string& query,
        std::size_t top_k
    ) {
        // Seal the index before allowing concurrent readers.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            sealed_ = true;
        }

        // Acquire exclusive ownership of a workspace.
        std::size_t slot = 0;

        {
            std::unique_lock<std::mutex> lock(workspace_mutex_);

            workspace_cv_.wait(lock, [this]() {
                for (bool busy : workspace_busy_) {
                    if (!busy) {
                        return true;
                    }
                }
                return false;
            });

            for (std::size_t i = 0; i < workspace_busy_.size(); ++i) {
                if (!workspace_busy_[i]) {
                    workspace_busy_[i] = true;
                    slot = i;
                    break;
                }
            }
        }

        // Return the workspace even if BM25 search throws.
        struct WorkspaceGuard {
            PythonSearchEngine* owner;
            std::size_t slot;

            ~WorkspaceGuard() {
                {
                    std::lock_guard<std::mutex> lock(
                        owner->workspace_mutex_
                    );
                    owner->workspace_busy_[slot] = false;
                }

                owner->workspace_cv_.notify_one();
            }
        };

        WorkspaceGuard guard{this, slot};

        return engine_.search_dense(
            query,
            top_k,
            workspaces_[slot]
        );
    }


    py::dict workspace_memory_stats() {
        std::lock_guard<std::mutex> lock(workspace_mutex_);

        py::list workspaces;
        std::size_t total_allocated_bytes = 0;
        std::size_t busy_count = 0;

        for (std::size_t i = 0; i < WORKSPACE_COUNT; ++i) {
            py::dict item;
            item["workspace_id"] = i;

            if (workspace_busy_[i]) {
                item["busy"] = true;
                item["scores_capacity"] = py::none();
                item["generations_capacity"] = py::none();
                item["touched_docs_capacity"] = py::none();
                item["allocated_bytes"] = py::none();

                ++busy_count;
            } else {
                const auto& workspace = workspaces_[i];

                const std::size_t scores_bytes =
                    workspace.scores.capacity() * sizeof(double);

                const std::size_t generations_bytes =
                    workspace.generations.capacity() * sizeof(uint32_t);

                const std::size_t touched_docs_bytes =
                    workspace.touched_docs.capacity() * sizeof(uint32_t);

                const std::size_t allocated_bytes =
                    scores_bytes +
                    generations_bytes +
                    touched_docs_bytes;

                total_allocated_bytes += allocated_bytes;

                item["busy"] = false;
                item["scores_size"] = workspace.scores.size();
                item["scores_capacity"] = workspace.scores.capacity();
                item["generations_capacity"] =
                    workspace.generations.capacity();
                item["touched_docs_size"] =
                    workspace.touched_docs.size();
                item["touched_docs_capacity"] =
                    workspace.touched_docs.capacity();
                item["allocated_bytes"] = allocated_bytes;
            }

            workspaces.append(item);
        }

        py::dict result;
        result["workspace_count"] = WORKSPACE_COUNT;
        result["busy_count"] = busy_count;
        result["complete"] = (busy_count == 0);
        result["workspaces"] = workspaces;

        if (busy_count == 0) {
            result["total_allocated_bytes"] = total_allocated_bytes;
        } else {
            result["total_allocated_bytes"] = py::none();
        }

        return result;
    }

    std::size_t document_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return engine_.document_count();
    }

    std::size_t vocabulary_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return engine_.vocabulary_size();
    }

    std::size_t posting_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return engine_.posting_count();
    }

private:
    static constexpr std::size_t WORKSPACE_COUNT = 4;

    retrieval::SearchEngine engine_;

    std::array<
        retrieval::SearchWorkspace,
        WORKSPACE_COUNT
    > workspaces_;

    std::array<bool, WORKSPACE_COUNT> workspace_busy_{};

    mutable std::mutex mutex_;
    std::mutex workspace_mutex_;
    std::condition_variable workspace_cv_;

    bool sealed_ = false;
};

}  // namespace

PYBIND11_MODULE(retrieval_cpp, m) {
    m.doc() = "C++ BM25 retrieval engine Python bindings";

    py::class_<retrieval::SearchResult>(
        m,
        "SearchResult"
    )
        .def_readonly(
            "doc_id",
            &retrieval::SearchResult::doc_id
        )
        .def_readonly(
            "score",
            &retrieval::SearchResult::score
        );

    py::class_<PythonSearchEngine>(
        m,
        "SearchEngine"
    )
        .def(py::init<>())

        .def(
            "load_corpus",
            &PythonSearchEngine::load_corpus,
            py::arg("path")
        )

        .def(
            "add_document",
            &PythonSearchEngine::add_document,
            py::arg("doc_id"),
            py::arg("text")
        )

        .def(
            "search",
            &PythonSearchEngine::search,
            py::arg("query"),
            py::arg("top_k") = 10
        )

        .def(
            "search_dense",
            &PythonSearchEngine::search_dense,
            py::arg("query"),
            py::arg("top_k") = 10,
            py::call_guard<py::gil_scoped_release>()
        )

        .def(
            "workspace_memory_stats",
            &PythonSearchEngine::workspace_memory_stats
        )

        .def(
            "document_count",
            &PythonSearchEngine::document_count
        )

        .def(
            "vocabulary_size",
            &PythonSearchEngine::vocabulary_size
        )

        .def(
            "posting_count",
            &PythonSearchEngine::posting_count
        );
}

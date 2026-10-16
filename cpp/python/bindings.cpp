#include <cstddef>
#include <cstdint>
#include <memory>
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
        std::lock_guard<std::mutex> lock(mutex_);
        sealed_ = true;

        // Workspace is local to this search.
        // This avoids sharing mutable accumulator state.
        return engine_.search_dense(
            query,
            top_k,
            workspace_
        );
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
    retrieval::SearchEngine engine_;
    retrieval::SearchWorkspace workspace_;
    mutable std::mutex mutex_;
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

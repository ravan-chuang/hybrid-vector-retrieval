#pragma once

#include "retrieval/search_engine.hpp"

#include <cstddef>
#include <string>

namespace retrieval {

std::size_t load_corpus(
    const std::string& path,
    SearchEngine& engine
);

}

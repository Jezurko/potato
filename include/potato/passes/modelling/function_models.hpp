#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/YAMLParser.h>
#include <llvm/Support/YAMLTraits.h>
POTATO_UNRELAX_WARNINGS

#include "potato/util/common.hpp"
#include <vector>

namespace potato::models {
    enum class arg_effect { none, alloc, static_alloc, deref_alloc, realloc_ptr, realloc_res, src, deref_src, copy_trg, assign_trg, unknown };
    enum class ret_effect { none, alloc, static_alloc, realloc_res, copy_trg, assign_trg, unknown};

    struct function_model {
        ret_effect ret;
        std::vector< arg_effect > args;
    };

    struct named_function_model {
        std::string name;
        llvm::SmallVector< function_model, 2 > models;
    };

    using function_models = llvm::StringMap< llvm::SmallVector< function_model, 2 > >;

    function_models load_and_parse(string_ref config);
} // namespace potato::models

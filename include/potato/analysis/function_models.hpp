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

namespace potato::analysis {
    enum class arg_effect { none, alloc, copy_src, copy_trg, assign_src, assign_trg, unknown };
    enum class ret_effect { none, alloc, copy_trg, assign_trg, unknown};

    struct function_model {
        ret_effect ret;
        std::vector< arg_effect > args;
    };

    struct named_function_model {
        std::string name;
        function_model model;
    };

    using function_models = llvm::StringMap< function_model >;

    function_models load_and_parse(string_ref config);
} // namespace potato::analysis

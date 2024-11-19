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
    enum class arg_effect { none, alloc, copy_src, copy_trg, unknown };
    enum class ret_effect { none, alloc, copy_trg, unknown};

    struct function_model {
        ret_effect ret;
        std::vector< arg_effect > args;
    };

    struct named_function_model {
        std::string name;
        function_model model;
    };

    using function_models = llvm::StringMap< function_model >;
} // namespace potato::analysis

LLVM_YAML_IS_SEQUENCE_VECTOR(potato::analysis::arg_effect);
LLVM_YAML_IS_SEQUENCE_VECTOR(potato::analysis::named_function_model);

using llvm::yaml::IO;
using llvm::yaml::MappingTraits;
using llvm::yaml::ScalarEnumerationTraits;

template<>
struct ScalarEnumerationTraits< potato::analysis::arg_effect >
{
    static void enumeration(IO &io, potato::analysis::arg_effect &value) {
        io.enumCase(value, "none", potato::analysis::arg_effect::none);
        io.enumCase(value, "alloc", potato::analysis::arg_effect::alloc);
        io.enumCase(value, "copy_trg", potato::analysis::arg_effect::copy_trg);
        io.enumCase(value, "copy_src", potato::analysis::arg_effect::copy_src);
        io.enumCase(value, "unknown", potato::analysis::arg_effect::unknown);
    }
};

template<>
struct ScalarEnumerationTraits< potato::analysis::ret_effect >
{
    static void enumeration(IO &io, potato::analysis::ret_effect &value) {
        io.enumCase(value, "none", potato::analysis::ret_effect::none);
        io.enumCase(value, "alloc", potato::analysis::ret_effect::alloc);
        io.enumCase(value, "copy_trg", potato::analysis::ret_effect::copy_trg);
        io.enumCase(value, "unknown", potato::analysis::ret_effect::unknown);
    }
};

template<>
struct MappingTraits< potato::analysis::function_model >
{
    static void mapping(IO &io, potato::analysis::function_model &model) {
        io.mapRequired("return_effect", model.ret);
        io.mapRequired("arguments", model.args);
    }
};

template <>
struct MappingTraits< potato::analysis::named_function_model > {
    static void mapping(IO &io, potato::analysis::named_function_model &model) {
        io.mapRequired("function", model.name);
        io.mapRequired("model", model.model);
    }
};

namespace potato::analysis {
    function_models load_and_parse(string_ref config);
} // namespace potato::analysis

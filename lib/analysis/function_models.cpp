#include "potato/analysis/function_models.hpp"

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
        io.enumCase(value, "realloc_ptr", potato::analysis::arg_effect::realloc_ptr);
        io.enumCase(value, "realloc_res", potato::analysis::arg_effect::realloc_res);
        io.enumCase(value, "copy_trg", potato::analysis::arg_effect::copy_trg);
        io.enumCase(value, "src", potato::analysis::arg_effect::src);
        io.enumCase(value, "assign_trg", potato::analysis::arg_effect::assign_trg);
        io.enumCase(value, "deref_src", potato::analysis::arg_effect::deref_src);
        io.enumCase(value, "unknown", potato::analysis::arg_effect::unknown);
    }
};

template<>
struct ScalarEnumerationTraits< potato::analysis::ret_effect >
{
    static void enumeration(IO &io, potato::analysis::ret_effect &value) {
        io.enumCase(value, "none", potato::analysis::ret_effect::none);
        io.enumCase(value, "alloc", potato::analysis::ret_effect::alloc);
        io.enumCase(value, "realloc_res", potato::analysis::ret_effect::realloc_res);
        io.enumCase(value, "copy_trg", potato::analysis::ret_effect::copy_trg);
        io.enumCase(value, "assign_trg", potato::analysis::ret_effect::assign_trg);
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
    function_models load_and_parse(string_ref config) {
        function_models models;
        auto file_or_err = llvm::MemoryBuffer::getFile(config);
        if (auto ec = file_or_err.getError()) {
            llvm::errs() << "Could not open config file: " << ec.message() << "\n";
            assert(false);
        }

        std::vector< named_function_model > functions;

        llvm::yaml::Input yin(file_or_err.get()->getBuffer());
        yin >> functions;

        if (yin.error()) {
            llvm::errs() << "Error parsing config file: " << yin.error().message() << "\n";
            assert(false);
        }

        for (auto &&named : functions) {
            models.insert_or_assign(std::move(named.name), std::move(named.model));
        }
        return models;
    }
} // namespace potato::analysis

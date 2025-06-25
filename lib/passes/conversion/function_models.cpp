#include "potato/passes/modelling/function_models.hpp"

LLVM_YAML_IS_SEQUENCE_VECTOR(potato::models::arg_effect);
LLVM_YAML_IS_SEQUENCE_VECTOR(potato::models::function_model);
LLVM_YAML_IS_SEQUENCE_VECTOR(potato::models::named_function_model);

using llvm::yaml::IO;
using llvm::yaml::MappingTraits;
using llvm::yaml::ScalarEnumerationTraits;

template<>
struct ScalarEnumerationTraits< potato::models::arg_effect >
{
    static void enumeration(IO &io, potato::models::arg_effect &value) {
        io.enumCase(value, "none", potato::models::arg_effect::none);
        io.enumCase(value, "alloc", potato::models::arg_effect::alloc);
        io.enumCase(value, "static_alloc", potato::models::arg_effect::static_alloc);
        io.enumCase(value, "deref_alloc", potato::models::arg_effect::static_alloc);
        io.enumCase(value, "realloc_ptr", potato::models::arg_effect::realloc_ptr);
        io.enumCase(value, "src", potato::models::arg_effect::src);
        io.enumCase(value, "assign_trg", potato::models::arg_effect::assign_trg);
        io.enumCase(value, "deref_src", potato::models::arg_effect::deref_src);
    }
};

template<>
struct ScalarEnumerationTraits< potato::models::ret_effect >
{
    static void enumeration(IO &io, potato::models::ret_effect &value) {
        io.enumCase(value, "none", potato::models::ret_effect::none);
        io.enumCase(value, "alloc", potato::models::ret_effect::alloc);
        io.enumCase(value, "static_alloc", potato::models::ret_effect::static_alloc);
        io.enumCase(value, "realloc_res", potato::models::ret_effect::realloc_res);
        io.enumCase(value, "copy_trg", potato::models::ret_effect::copy_trg);
        io.enumCase(value, "unknown", potato::models::ret_effect::unknown);
    }
};

template<>
struct MappingTraits< potato::models::function_model >
{
    static void mapping(IO &io, potato::models::function_model &model) {
        io.mapRequired("return_effect", model.ret);
        io.mapRequired("arguments", model.args);
    }
};

template <>
struct MappingTraits< potato::models::named_function_model > {
    static void mapping(IO &io, potato::models::named_function_model &model) {
        io.mapRequired("function", model.name);
        io.mapRequired("model", model.models);
    }
};

namespace potato::models {
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
            models.insert_or_assign(std::move(named.name), std::move(named.models));
        }
        return models;
    }
} // namespace potato::models

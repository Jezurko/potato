#include "potato/analysis/function_models.hpp"

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

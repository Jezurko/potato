#include "potato/analysis/utils.hpp"

namespace potato::analysis {

void pt_element::print(llvm::raw_ostream &os) const {
    switch (kind) {
        case elem_kind::alloca:
            os << "mem_alloc at: " << operation->getLoc();
            if (val) {
                os << " for: " << val;
            }
            break;
        case elem_kind::var:
            os << "var: " << val;
        break;
        case elem_kind::func: {
                auto symbol = mlir::cast< mlir::SymbolOpInterface >(operation);
                os << symbol.getName();
            break;
        }
        case elem_kind::named_var:
            auto symbol = mlir::cast< mlir::SymbolOpInterface >(operation);
            os << symbol.getName() << "at" << symbol->getLoc();
        break;
    }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const pt_element &e) {
    e.print(os);
    return os;
}

} // namespace potato::analysis

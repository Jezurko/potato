#include "potato/analysis/pt.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {
mlir_loc mem_loc_anchor::getLoc() const { return getValue().first->getLoc(); }

void mem_loc_anchor::print(llvm::raw_ostream &os) const {
    os << "mem_alloc at: " << getLoc();
    if (getValue().second != 0)
        os << "with unique id " << getValue().second;
}

mlir_loc named_val_anchor::getLoc() const { return getValue()->getLoc(); }

void named_val_anchor::print(llvm::raw_ostream &os) const {
    if (auto symbol = mlir::dyn_cast< symbol_iface >(getValue()))
        os << symbol.getName();
    else
        os << "named_val: " << *getValue();
}

mlir_loc var_arg_anchor::getLoc() const { return getValue().first->getLoc(); }

void var_arg_anchor::print(llvm::raw_ostream &os) const {
    os << "vararg anchor";
    if (auto func = mlir::dyn_cast< func_iface >(getValue().first))
        os << " for: " << func.getName();
    os << " at: " << getLoc();
    if (getValue().second != 0)
        os << "with unique id " << getValue().second;
}
} // namespace potato::analysis

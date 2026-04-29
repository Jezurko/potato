#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/SetOperations.h>
#include <mlir/IR/SymbolTable.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/andersen_bv.hpp"
#include "potato/analysis/pt.hpp"
#include "potato/analysis/utils.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/dialect/types.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {
anchor_index::anchor_index(llvm::SmallVector< lattice_anchor > anchor_list) : anchor_list(std::move(anchor_list)) {
    for (auto &&[idx, anchor] : llvm::enumerate(anchor_list))
        anchor_to_index.try_emplace(anchor, idx);
}

logical_result aa_bv_analysis::initialize(mlir_operation *root) {
    llvm::SmallVector< lattice_anchor > anchor_list;
    root->walk([&](mlir_operation *op) {
        if (mlir::isa< mlir::SymbolOpInterface >(op)) {
            anchor_list.push_back(
                getLatticeAnchor< named_val_anchor >(op)
            );
            if (auto fn = mlir::dyn_cast< func_iface >(op)) {
                if (auto fn_type = mlir::dyn_cast< pt::FunctionType >(fn.getFunctionType())) {
                    if (fn_type.isVarArg())
                        anchor_list.push_back(getLatticeAnchor< var_arg_anchor >(op));
                }
            }
        }
        if (mlir::isa< pt::AllocOp >(op)) {
            anchor_list.push_back(
                getLatticeAnchor< mem_loc_anchor >(op)
            );
        }
    });
    index = anchor_index(std::move(anchor_list));
    return base::initialize(root);
}
} // namespace potato::analysis

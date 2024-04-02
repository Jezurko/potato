#include "potato/dialect/analysis/utils.hpp"

namespace potato::analysis {
auto get_args(ppoint &point) -> mlir_block::BlockArgListType {
    if (auto block = mlir::dyn_cast< mlir_block * >(point))
        return block->getArguments();
    return {};
}
} // namespace potato::analysis

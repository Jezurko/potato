#include "potato/analysis/steensgaard.hpp"
namespace potato::analysis {

unsigned int steensgaard::alloc_count() {
    return info->mem_loc_count++;
}

llvm::StringRef steensgaard::get_alloc_name() {
    if (!alloc_name)
        alloc_name = "mem_alloc" + std::to_string(alloc_count());
    return alloc_name.value();
}

} // namespace potato::analysis

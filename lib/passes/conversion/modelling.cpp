#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/ops.hpp"
#include "potato/dialect/types.hpp"
#include "potato/passes/conversions.hpp"
#include "potato/passes/modelling/config.hpp"
#include "potato/passes/modelling/function_models.hpp"
#include "potato/util/common.hpp"

namespace potato::conv::modelling
{
    struct FunctionModellingPass : FunctionModellingBase< FunctionModellingPass >
    {
        static bool is_inlineable(mlir::ArrayRef< models::function_model > models) {
            for (const auto &model : models) {
                if (model.ret == models::ret_effect::static_alloc)
                    return false;
                for (const auto &effect : model.args) {
                    if (effect == models::arg_effect::static_alloc)
                        return false;
                }
            }
            return true;
        }

        static mlir_operation *create_body(
                value_range args, mlir_loc loc, op_builder &builder,
                mlir::ArrayRef< models::function_model > models
        ) {
            llvm::SmallVector< mlir_value, 2 > reallocated;
            llvm::SmallVector< mlir_value, 1 > src;
            llvm::SmallVector< mlir_value, 1 > assign_trgs;
            mlir_operation *ret_op = nullptr;
            mlir_type ptr_type = pt::PointerType::get(builder.getContext());
            for (const auto &model : models) {
                for (const auto &[effect, arg] : llvm::zip(model.args, args)) {
                    using arg_effect = models::arg_effect;
                    switch (effect) {
                        case arg_effect::none:
                            continue;
                        case arg_effect::alloc:
                        case arg_effect::static_alloc: {
                            auto alloc = builder.create< pt::AllocOp >(loc, ptr_type);
                            builder.create< pt::AssignOp >(loc, arg, alloc);
                            break;
                        }
                        case arg_effect::realloc_ptr: {
                            auto alloc = builder.create< pt::AllocOp >(loc, ptr_type);
                            reallocated.push_back(arg);
                            reallocated.push_back(alloc);
                            break;
                        }
                        case arg_effect::src:
                            src.push_back(arg);
                            break;
                        case arg_effect::deref_src: {
                            auto deref = builder.create< pt::DereferenceOp >(loc, ptr_type, arg);
                            src.push_back(deref);
                            break;
                        }
                        case arg_effect::assign_trg: {
                            // We might not yet have all sources
                            // Collect targets and resolve later
                            assign_trgs.push_back(arg);
                            break;
                        }
                    }
                }
            }
            if (!assign_trgs.empty()) {
                mlir_value copy_val;
                if (src.size() == 1)
                    copy_val = src[0];
                else
                    copy_val = builder.create< pt::CopyOp >(loc, ptr_type, src);
                for (const auto &trg : assign_trgs) {
                    builder.create< pt::AssignOp >(loc, trg, copy_val);
                }
            }
            for (const auto &model : models) {
                using ret_effect = models::ret_effect;
                switch (model.ret) {
                    case ret_effect::none:
                        break;
                    case ret_effect::alloc:
                    case ret_effect::static_alloc:
                        ret_op = builder.create< pt::AllocOp >(loc, ptr_type);
                        break;
                    case ret_effect::realloc_res:
                        ret_op = builder.create< pt::CopyOp >(loc, ptr_type, reallocated);
                        break;
                    case ret_effect::copy_trg:
                        ret_op = builder.create< pt::CopyOp >(loc, ptr_type, src);
                        break;
                    case ret_effect::unknown:
                        ret_op = builder.create< pt::UnknownPtrOp >(loc, ptr_type);
                }
            }
            return ret_op;
        }

        void model_function(func_iface fn, op_builder &builder, const models::function_models &models) {
            auto model_it = models.find(fn.getName());
            if (model_it == models.end()) {
                llvm::errs() << "External function without a model: " << fn.getName() << "\n";
                // TODO: Add pass option for hard fail on unknown function
                return;
            }
            mlir::ArrayRef< models::function_model > fn_models = model_it->getValue();

            if (inline_bodies) {
                if (is_inlineable(fn_models)) {
                    builder.setInsertionPoint(fn);
                }
            }

            auto entry = fn.addEntryBlock();
            builder.setInsertionPointToStart(entry);
            if (auto ret_op = create_body(entry->getArguments(), fn.getLoc(), builder, fn_models)) {
                builder.create< pt::YieldOp >(fn.getLoc(), ret_op->getResults());
            } else {
                value_range results{};
                if (fn.getNumResults() != 0)
                    results = {builder.create< pt::ConstantOp >(fn.getLoc(), pt::PointerType::get(builder.getContext()))};
                builder.create< pt::YieldOp >(fn.getLoc(), results);
            }
        }

        bool maybe_inline(
                func_iface fn, mlir::SymbolTable &symbol_table, mlir::SymbolUserMap &map,
                op_builder &builder, const models::function_models &models
        ) {
            auto model_it = models.find(fn.getName());
            if (model_it == models.end()) {
                llvm::errs() << "External function without a model: " << fn.getName() << "\n";
                // TODO: Add pass option for hard fail on unknown function
                return false;
            }
            mlir::ArrayRef< models::function_model > fn_models = model_it->getValue();

            for (auto user : map.getUsers(fn)) {
                builder.setInsertionPoint(user);
                auto new_result_op = create_body(user->getOperands(), user->getLoc(), builder, fn_models);
                auto old_results = user->getResults();
                for (auto [new_res, old_res] : llvm::zip(new_result_op->getResults(), old_results)) {
                    old_res.replaceAllUsesWith(new_res);
                }
                user->erase();
            }
            symbol_table.erase(fn);
            return true;
        }

        void runOnOperation() override {

            auto root = getOperation();

            auto &mctx = getContext();
            auto builder = mlir::OpBuilder(&mctx);

            std::unique_ptr< mlir::SymbolTableCollection > table_collection{};
            std::unique_ptr< mlir::SymbolUserMap > symbol_map{};

            if (inline_bodies) {
                table_collection = std::make_unique< mlir::SymbolTableCollection >();
                table_collection->getSymbolTable(root);
                symbol_map = std::make_unique< mlir::SymbolUserMap >(*table_collection.get(), root);
            }

            auto models = models::load_and_parse(models::pointsto_analysis_config);
            for (auto &op : root) {
                if (auto fn = mlir::dyn_cast< func_iface >(op)) {
                    if (fn.isExternal()) {
                        if (inline_bodies) {
                            assert(table_collection && symbol_map &&
                                   "inlining without symbol info!");

                            auto symbol_table = table_collection->getSymbolTable(root);
                            if (maybe_inline(fn, symbol_table, *symbol_map.get(), builder, models))
                                continue;
                        }
                        model_function(fn, builder, models);
                    }
                }
            }
        }
    };
} // namespace potato::conv::modelling

std::unique_ptr< mlir::Pass > potato::createFunctionModellingPass() {
    return std::make_unique< potato::conv::modelling::FunctionModellingPass >();
}

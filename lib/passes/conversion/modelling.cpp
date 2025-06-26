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
        mlir_value create_body(value_range args, mlir_loc loc, op_builder &builder, mlir::ArrayRef< models::function_model > models) {
            llvm::SmallVector< mlir_value, 2 > reallocated;
            llvm::SmallVector< mlir_value, 1 > src;
            mlir_value ret_val;
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
                            mlir_value copy_val;
                            if (src.size() == 1)
                                copy_val = src[0];
                            else
                                copy_val = builder.create< pt::CopyOp >(loc, ptr_type, src);
                            builder.create< pt::AssignOp >(loc, arg, copy_val);
                            break;
                        }
                    }
                }
                using ret_effect = models::ret_effect;
                switch (model.ret) {
                    case ret_effect::none:
                        break;
                    case ret_effect::alloc:
                    case ret_effect::static_alloc:
                        ret_val = builder.create< pt::AllocOp >(loc, ptr_type);
                        break;
                    case ret_effect::realloc_res:
                        ret_val = builder.create< pt::CopyOp >(loc, ptr_type, reallocated);
                        break;
                    case ret_effect::copy_trg:
                        ret_val = builder.create< pt::CopyOp >(loc, ptr_type, src);
                        break;
                    case ret_effect::unknown:
                        ret_val = builder.create< pt::UnknownPtrOp >(loc, ptr_type);
                }
            }
            return ret_val;
        }

        void model_function(func_iface fn, op_builder &builder, const models::function_models &models) {
            auto model_it = models.find(fn.getName());
            if (model_it == models.end()) {
                llvm::errs() << "External function without a model: " << fn.getName() << "\n";
                // TODO: Add pass option for hard fail on unknown function
                return;
            }
            mlir::ArrayRef< models::function_model > fn_models = model_it->getValue();

            if (inline_bodies) { /*TODO*/ }

            auto entry = fn.addEntryBlock();
            builder.setInsertionPointToStart(entry);
            if (auto ret_val = create_body(entry->getArguments(), fn.getLoc(), builder, fn_models))
                builder.create< pt::YieldOp >(fn.getLoc(), ret_val);
            else
                builder.create< pt::YieldOp >(fn.getLoc(), value_range());
        }
        // void inlineFunction

        void runOnOperation() override {
            if (inline_bodies) {
                llvm::errs() << "Inlining NYI for modelling pass\n";
                return signalPassFailure();
            }

            auto root = getOperation();

            auto &mctx = getContext();
            auto builder = mlir::OpBuilder(&mctx);

            auto models = models::load_and_parse(models::pointsto_analysis_config);
            for (auto &op : root) {
                if (auto fn = mlir::dyn_cast< func_iface >(op)) {
                    if (fn.isExternal()) {
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

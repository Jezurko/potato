#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/CallGraph.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlow/SparseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/TypeSwitch.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/ops.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {

// Memory location is identified by the allocating operation first.
// Multiple unique allocations performed by the same operation can be modelled
// using a uniquer id.
struct mem_loc : std::pair< mlir_operation *, size_t > {
    using pair_t = std::pair< mlir_operation *, size_t >;
    mem_loc(mlir_operation *op) : pair_t(op, 0) {}

    friend struct ::llvm::DenseMapInfo< potato::analysis::mem_loc >;
private:
    mem_loc(pair_t pair) : pair_t(pair) {}
};

} // namespace potato::analysis

template<>
struct llvm::DenseMapInfo< potato::analysis::mem_loc > {
    using pair_info = llvm::DenseMapInfo< std::pair< mlir_operation *, size_t > >;
    using loc = potato::analysis::mem_loc;

    static inline loc getEmptyKey() { return pair_info::getEmptyKey(); };
    static inline loc getTombstoneKey() { return pair_info::getTombstoneKey(); };
    static inline unsigned getHashValue(const loc& loc) { return pair_info::getHashValue(loc); }
    static bool isEqual(const loc& lhs, const loc& rhs) { return pair_info::isEqual(lhs, rhs); }
};

namespace potato::analysis {

struct mem_loc_anchor : mlir::GenericLatticeAnchorBase< mem_loc_anchor, mem_loc > {
    using Base::Base;

    mlir_operation *getAllocator() const { return getValue().first; }
    size_t getUniquer() const { return getValue().second; }

    mlir_loc getLoc() const override;
    void print(llvm::raw_ostream &) const override;
};

struct named_val_anchor : mlir::GenericLatticeAnchorBase< named_val_anchor, mlir_operation * > {
    using Base::Base;

    mlir_loc getLoc() const override;
    void print(llvm::raw_ostream &) const override;
};

template< typename derived_t >
struct pt_lattice_base : mlir::AnalysisState {

    pt_lattice_base(lattice_anchor anchor) : AnalysisState(anchor) {}
    pt_lattice_base(mlir_value value) : AnalysisState(value) {}

    void onUpdate(mlir::DataFlowSolver *solver) const override {
        AnalysisState::onUpdate(solver);
        if (auto value = mlir::dyn_cast< mlir_value >(anchor))
            for (mlir_operation *user : value.getUsers())
                for (dfa *analysis : use_def_subs)
                   solver->enqueue({solver->getProgramPointAfter(user), analysis});
        for (auto dep : extra_deps)
            for (dfa *analysis : use_def_subs)
                solver->enqueue({solver->getProgramPointAfter(dep), analysis});
    }

    // Subscribe an analysis (including the pt analysis itself) to the updates
    // of this lattice. If the lattice is updated, the registered analyses are
    // invoked on the users of the anchor value
    void use_def_subscribe(dfa *analysis) { use_def_subs.insert(analysis); }

    // Add extra users. This is useful for e.g. adding dereference op
    // as a dependency to members of arguments points-to set.
    // Example:
    // %ptr -> {x}, x -> {y}
    // %set = deref %ptr
    // In this case %set -> {y} and if the points-to set of x gets updated
    // it has to be updated as well
    void add_user(mlir_operation *op) { extra_deps.insert(op); }
private:
    llvm::SetVector< dfa *, llvm::SmallVector< dfa *, 4 >, llvm::SmallPtrSet< dfa *, 4 > >
        use_def_subs;
    llvm::SmallPtrSet< mlir_operation *, 8 > extra_deps;
};

// Heavily inspired by (Abstract)SparseForwardDataFlowAnalysis.

template< typename derived_t, typename lattice_t >
struct pt_analysis : dfa {
    using pt_lattice = lattice_t;
    using base = dfa;
    using const_lattices_ref = llvm::ArrayRef< const pt_lattice * >;
    using lattices_ref = llvm::ArrayRef< pt_lattice * >;
    friend lattice_t;

protected:

    derived_t &derived() { return static_cast< derived_t & >(*this); }
    const derived_t &derived() const { return static_cast< const derived_t & >(*this); }

private:

    ///////////////////////////////////////////////////////////////////////////////////////
    // Functions that are user modifiable should be called through private _impl methods //
    ///////////////////////////////////////////////////////////////////////////////////////

    // Default implementation customization points:

    void visit_external_call_impl(
        mlir::CallOpInterface call, const_lattices_ref arg_lattices, lattices_ref res_lattices
    ) {
        derived().visit_external_call(call, arg_lattices, res_lattices);
    }

    void visit_non_control_flow_arguments_impl(
        mlir_operation *op, const mlir::RegionSuccessor &successor,
        lattices_ref arg_lattices, unsigned first_index
    ) {
        return derived().visit_non_control_flow_arguments(
            op, successor, arg_lattices, first_index
        );
    }

    logical_result visit_call_operation_impl(
        mlir::CallOpInterface call,
        const_lattices_ref operand_lattices,
        lattices_ref result_lattices
    ) {
        return derived().visit_call_operation(call, operand_lattices, result_lattices);
    }

    logical_result visit_fptr_call_impl(
        mlir::CallOpInterface call,
        const_lattices_ref operand_lattices,
        lattices_ref result_lattices
    ) {
        return derived().visit_fptr_call(call, operand_lattices, result_lattices);
    }

    constexpr bool add_deps_impl() {
        return derived().add_deps();
    }

    logical_result visit_operation_impl(
        mlir_operation *op, const_lattices_ref operand_lattices, lattices_ref res_lattices
    ) {
        return derived().visit_operation(op, operand_lattices, res_lattices);
    }

    logical_result visit_pt_op_impl(
        auto op, const_lattices_ref operand_lattices, lattices_ref res_lattices
    ) {
        return derived().visit_pt_op(op, operand_lattices, res_lattices);
    }

    logical_result visit_non_pt_op_impl(
        mlir_operation *op, const_lattices_ref operand_lattices, lattices_ref res_lattices
    ) {
        return derived().visit_non_pt_op(op, operand_lattices, res_lattices);
    }

    logical_result visit_unrealized_cast_impl(
        mlir::UnrealizedConversionCastOp op,
        const_lattices_ref operand_lattices, lattices_ref res_lattices
    ) {
        return derived().visit_unrealized_cast(op, operand_lattices, res_lattices);
    }

    logical_result visit_branch_impl(
        branch_iface op, const_lattices_ref operand_lattices, lattices_ref res_lattices
    ) {
        return derived().visit_branch(op, operand_lattices, res_lattices);
    }

    // derived_t has to provide a custom definition for the following methods:

    void set_to_entry_state_impl(pt_lattice *lattice) {
        return derived().set_to_entry_state(lattice);
    }

    void register_anchors_impl() {
        return derived().register_anchors();
    }

public:

    explicit pt_analysis(mlir::DataFlowSolver &solver)
        : dfa(solver) {
           // Do I want to require this?
           registerAnchorKind< mlir::dataflow::CFGEdge >();
           registerAnchorKind< mem_loc_anchor >();
           register_anchors_impl();
    }

    logical_result initialize(mlir_operation *root) override {
        for (mlir_region &region : root->getRegions()) {
            if (region.empty())
                continue;
            for (mlir_value argument : region.front().getArguments())
                set_to_entry_state_impl(get_lattice_element(argument));
        }
        tables.getSymbolTable(root);
        return initialize_recursively(root);
    };

    logical_result visit(ppoint point) override {
        if (!point->isBlockStart())
          return visit_operation(point->getPrevOp());
        visit_block(point->getBlock());
        return mlir::success();
    };

    logical_result initialize_recursively(mlir_operation *op) {
        if (failed(visit_operation(op)))
            return mlir::failure();

        for (mlir_region &region : op->getRegions()) {
            for (mlir_block &block : region) {
                getOrCreate<mlir::dataflow::Executable>(
                    getProgramPointBefore(&block))->blockContentSubscribe(this);
                visit_block(&block);
            }
        }
        return mlir::success();
    }

    constexpr bool add_deps() { return true; }

    logical_result visit_non_pt_op(mlir_operation *, const_lattices_ref, lattices_ref) {
        return mlir::success();
    }

    void visit_external_call(
        mlir::CallOpInterface call, const_lattices_ref arg_lattices, lattices_ref res_lattices
    ) {
        set_all_to_entry_state(res_lattices);
    }

    llvm::SmallVector< mlir::CallableOpInterface > fptr_to_callables(mlir_value fptr) {
        auto fptr_pointees = getOrCreate< pt_lattice >(fptr)->get_pointees();
        llvm::SmallVector< mlir::CallableOpInterface > callables;
        callables.reserve(fptr_pointees.size());

        for (const auto &pointee : fptr_pointees) {
            auto pointee_pp = mlir::dyn_cast< ppoint >(pointee);
            if (!pointee_pp)
                continue;
            if (auto callable = mlir::dyn_cast_if_present< mlir::CallableOpInterface >(pointee_pp->getOperation()))
                callables.push_back(callable);
        }
        // TODO: Add possibility to error out on non-funciton pointee
        return callables;
    }

    logical_result visit_fptr_call(
        mlir::CallOpInterface call, const_lattices_ref arg_lattices, lattices_ref res_lattices
    ) {
        return mlir::failure();
        auto callable = mlir::dyn_cast< mlir_value >(call.getCallableForCallee());
        if (!callable)
            return mlir::failure();
        for (auto callable : derived().fptr_to_callables(callable)) {
            auto callable_body = callable.getCallableRegion();
            // TODO: Add optional error?
            if (!callable_body)
                continue;

            auto fn_args = callable_body->getArguments();
            for (auto &&[fn_arg, call_arg] : llvm::zip(fn_args, arg_lattices))
                join(getOrCreate< pt_lattice >(fn_arg), *call_arg);
            callable_body->walk([&](mlir_operation *op) {
                if (!op->hasTrait< mlir::OpTrait::ReturnLike >())
                    return;
                for (auto &&[call_res, fn_res] : llvm::zip(res_lattices, op->getOperands()))
                    join(call_res, *get_lattice_element_for(getProgramPointAfter(call), fn_res));
            });
        }
        return mlir::success();
    }

    logical_result visit_call_operation(
        mlir::CallOpInterface call,
        const_lattices_ref operand_lattices,
        lattices_ref result_lattices
    ) {
        auto callable_op = dyn_cast_if_present< mlir::CallableOpInterface >(
            call.resolveCallableInTable(&tables)
        );
        if (!getSolverConfig().isInterprocedural() ||
            (callable_op && !callable_op.getCallableRegion()))
        {
            visit_external_call_impl(call, operand_lattices, result_lattices);
            return mlir::success();
        }

        const auto predecessors = getOrCreateFor< mlir::dataflow::PredecessorState >(
            getProgramPointAfter(call), getProgramPointAfter(call)
        );
        if (mlir::isa< mlir_value >(call.getCallableForCallee()))
            return visit_fptr_call_impl(call, operand_lattices, result_lattices);

        // TODO: this should simply check that we know all return sites from the function
        // Check if it's true and if we need to modify this in any way
        if (!predecessors->allPredecessorsKnown()) {
            set_all_to_entry_state(result_lattices);
            return mlir::success();
        }

        for (mlir_operation *predecessor : predecessors->getKnownPredecessors()) {
            for (auto &&[operand, res_lattice] :
                llvm::zip(predecessor->getOperands(), result_lattices))
            {
                join(res_lattice, *get_lattice_element_for(getProgramPointAfter(call), operand));
            }
        }
        return mlir::success();
    }

    void visit_callable_operation(
        mlir::CallableOpInterface callable,
        lattices_ref arg_lattices
    ) {
        mlir_block *entry_block = &callable.getCallableRegion()->front();
        const auto *callsites = getOrCreateFor< mlir::dataflow::PredecessorState >(
            getProgramPointBefore(entry_block), getProgramPointAfter(callable));

        // NOTE: Here we diverge from sparse analysis and assume all callsites are known.
        // This is not ideal, but currently we support only WPA, so it isn't an issue.
        // TODO: Introduce option for partial program analysis.
        if (!getSolverConfig().isInterprocedural()) {
            set_all_to_entry_state(arg_lattices);
        }

        for (mlir_operation *callsite : callsites->getKnownPredecessors()) {
            auto call = cast< mlir::CallOpInterface >(callsite);
            for (auto [arg, lattice] : llvm::zip(call.getArgOperands(), arg_lattices))
                join(lattice, *get_lattice_element_for(getProgramPointBefore(entry_block), arg));
        }
    }

    void visit_non_control_flow_arguments(
        mlir_operation *op, const mlir::RegionSuccessor &successor,
        lattices_ref arg_lattices, unsigned first_index
    ) {
        set_all_to_entry_state(arg_lattices.take_front(first_index));
        set_all_to_entry_state(arg_lattices.drop_front(
            first_index + successor.getSuccessorInputs().size())
        );
    }

    pt_lattice *get_lattice_element(mlir_value value) {
        return getOrCreate< pt_lattice >(value);
    }

    const pt_lattice *get_lattice_element_for(ppoint point, mlir_value value) {
        pt_lattice *state = get_lattice_element(value);
        addDependency(state, point);
        return state;
    }

    void set_all_to_entry_state(lattices_ref lattices) {
        for (auto lattice : lattices)
            set_to_entry_state_impl(lattice);
    }

    void join(pt_lattice *lhs, const pt_lattice &rhs) {
        propagateIfChanged(lhs, lhs->join(rhs));
    }

    void join_all(pt_lattice *lhs, const_lattices_ref rhs) {
        auto changed = change_result::NoChange;
        for (auto lattice : rhs)
            changed |= lhs->join(*lattice);
        return propagateIfChanged(lhs, changed);
    }

    // Compared to SparseAnalysis we do not bail out on operations without results.
    // This is because e.g. assign(store) operations do not have result yet they do have
    // relevant points-to behaviour
    logical_result visit_operation(mlir_operation *op) {
        if (op->getBlock() != nullptr &&
            !getOrCreate< mlir::dataflow::Executable >(
                getProgramPointBefore(op->getBlock())
            )->isLive()
        ) {
            return mlir::success();
        }

        mlir::SmallVector< pt_lattice * > result_lattices;
        result_lattices.reserve(op->getNumResults());
        for (mlir_value result : op->getResults()) {
            pt_lattice *result_lattice = get_lattice_element(result);
            result_lattices.push_back(result_lattice);
        }

        if (auto branch = dyn_cast< mlir::RegionBranchOpInterface >(op)) {
            visit_region_successors(
                getProgramPointAfter(branch), branch,
                mlir::RegionBranchPoint::parent(), result_lattices
            );
            return mlir::success();
        }

        mlir::SmallVector< const pt_lattice * > operand_lattices;
        operand_lattices.reserve(op->getNumOperands());
        for (mlir_value operand : op->getOperands()) {
            pt_lattice *operand_lattice = get_lattice_element(operand);
            // TODO: point of customization!
            operand_lattice->use_def_subscribe(this);
        }
        if (auto call = dyn_cast< mlir::CallOpInterface >(op))
            return visit_call_operation_impl(call, operand_lattices, result_lattices);

        return visit_operation_impl(op, operand_lattices, result_lattices);
    }

    void visit_block(mlir_block *block) {
        if (block->getNumArguments() == 0)
            return;

        if (!getOrCreate<mlir::dataflow::Executable>(getProgramPointBefore(block))->isLive())
            return;
        mlir::SmallVector< pt_lattice * > arg_lattices;
        arg_lattices.reserve(block->getNumArguments());
        for (mlir::BlockArgument arg : block->getArguments()) {
            auto *arg_lattice = get_lattice_element(arg);
            arg_lattices.push_back(arg_lattice);
        }

        if (block->isEntryBlock()) {
            auto callable = mlir::dyn_cast< mlir::CallableOpInterface >(block->getParentOp());
            if (callable && callable.getCallableRegion() == block->getParent())
                return visit_callable_operation(callable, arg_lattices);

            if (auto branch = dyn_cast< mlir::RegionBranchOpInterface >(block->getParentOp())) {
                return visit_region_successors(
                    getProgramPointBefore(block), branch, block->getParent(), arg_lattices
                );
            }

            return visit_non_control_flow_arguments(
                block->getParentOp(), mlir::RegionSuccessor(block->getParent()), arg_lattices, 0
            );
        }

        for (auto it = block->pred_begin(), e = block->pred_end(); it != e; ++it) {
            mlir_block *pred = *it;
            auto *edge_executable = getOrCreate< mlir::dataflow::Executable >(
                getLatticeAnchor< mlir::dataflow::CFGEdge >(pred, block));

            edge_executable->blockContentSubscribe(this);
            if (!edge_executable->isLive())
                continue;

            if (auto branch = dyn_cast< branch_iface >(pred->getTerminator())) {
                auto operands = branch.getSuccessorOperands(it.getSuccessorIndex());
                for (auto [idx, lattice] : llvm::enumerate(arg_lattices)) {
                    if (auto operand = operands[idx]) {
                        join(lattice,
                            *get_lattice_element_for(getProgramPointBefore(block), operand)
                        );
                    } else {
                        // Conservatively consider internally produced arguments as entry points.
                        // TODO: is this necessary for us? when does this happen?
                        set_all_to_entry_state(lattice);
                    }
                }
            } else {
                return set_all_to_entry_state(arg_lattices);
            }
        }
    }

    void visit_region_successors(
        ppoint point, mlir::RegionBranchOpInterface branch,
        mlir::RegionBranchPoint successor,
        lattices_ref lattices)
    {
        const auto *preds = getOrCreateFor< mlir::dataflow::PredecessorState >(point, point);
        assert(preds->allPredecessorsKnown() && "unexpected unresolved region successors");

        for (mlir_operation *op : preds->getKnownPredecessors()) {
            std::optional< mlir::OperandRange > operands;

            if (op == branch) {
                operands = branch.getEntrySuccessorOperands(successor);
            } else if (auto region_terminator = mlir::dyn_cast< mlir::RegionBranchTerminatorOpInterface >(op)) {
                operands = region_terminator.getSuccessorOperands(successor);
            }

            // We can't reason about the data-flow
            if (!operands)
                return set_all_to_entry_state(lattices);

            mlir::ValueRange inputs = preds->getSuccessorInputs(op);
            assert(inputs.size() == operands->size() && "expected the same number of successor inputs as operands");

            unsigned first_idx = 0;
            if (inputs.size() != lattices.size()) {
                if (!point->isBlockStart()) {
                    if (!inputs.empty())
                        first_idx = mlir::cast< mlir::OpResult >(inputs.front()).getResultNumber();
                    visit_non_control_flow_arguments_impl(
                        branch,
                        mlir::RegionSuccessor(branch->getResults().slice(first_idx, inputs.size())),
                        lattices,
                        first_idx
                    );
                } else {
                    if (!inputs.empty())
                        first_idx = mlir::cast< mlir::BlockArgument >(inputs.front()).getArgNumber();
                    mlir_region *region = point->getBlock()->getParent();
                    visit_non_control_flow_arguments_impl(
                        branch,
                        mlir::RegionSuccessor(region, region->getArguments().slice(first_idx, inputs.size())),
                        lattices,
                        first_idx
                    );
                }

            }

            for (auto [operand, lattice] : llvm::zip(*operands, lattices.drop_front(first_idx)))
                join(lattice, *get_lattice_element_for(point, operand));
        }
    }

    logical_result visit_pt_op(pt::AddressOp op, const_lattices_ref operand_lts, lattices_ref res_lts) {
        auto symbol_op = tables.lookupNearestSymbolFrom(op.getOperation(), op.getSymbolAttr());
        auto symbol_anchor = getLatticeAnchor< named_val_anchor >(symbol_op);
        for (auto res_lat : res_lts)
            propagateIfChanged(res_lat, res_lat->insert(symbol_anchor));
        return mlir::success();
    }

    logical_result visit_pt_op(pt::AllocOp op, const_lattices_ref operand_lts, lattices_ref res_lts) {
        for (auto res_lat : res_lts)
            propagateIfChanged(res_lat, res_lat->insert(getLatticeAnchor< mem_loc_anchor >(op.getOperation())));
        return mlir::success();
    }

    logical_result visit_pt_op(pt::AssignOp, const_lattices_ref operand_lts, lattices_ref res_lts) {
        for (const auto &pointee : operand_lts[0]->get_pointees()) {
            join(getOrCreate< pt_lattice >(pointee), *operand_lts[1]);
        }
        return mlir::success();
    }

    logical_result visit_pt_op(pt::ConstantOp, const_lattices_ref operand_lts, lattices_ref res_lts) {
        return mlir::success();
    }

    logical_result visit_pt_op(pt::CopyOp, const_lattices_ref operand_lts, lattices_ref res_lts) {
        for (auto res_lat : res_lts)
            join_all(res_lat, operand_lts);
        return mlir::success();
    }

    logical_result visit_pt_op(pt::DereferenceOp, const_lattices_ref operand_lts, lattices_ref res_lts) {
        for (auto res_lat : res_lts) {
            for (auto operand_lat : operand_lts) {
                for (const auto &pointee : operand_lat->get_pointees()) {
                    join(res_lat, *getOrCreate< pt_lattice >(pointee));
                }
            }
        }
        return mlir::success();
    }

    logical_result visit_pt_op(pt::NamedVarOp op, const_lattices_ref operand_lts, lattices_ref res_lts) {
        auto &init = op.getInit();
        if (!init.empty()) {
            auto *ret_op = &init.back().back();
            if (ret_op->hasTrait< mlir::OpTrait::ReturnLike >()) {
                auto var_state =
                    getOrCreate< pt_lattice >(getLatticeAnchor< named_val_anchor >(op.getOperation()));
                for (auto ret_val : ret_op->getOperands())
                    join(var_state, *getOrCreate< pt_lattice >(ret_val));
            }
        }
        return mlir::success();
    }

    logical_result visit_pt_op(pt::UnknownPtrOp, const_lattices_ref operand_lts, lattices_ref res_lts) {
        for (auto res_lt : res_lts)
            propagateIfChanged(res_lt, res_lt->set_unknown());
        return mlir::success();
    }

    logical_result visit_unrealized_cast(mlir::UnrealizedConversionCastOp, const_lattices_ref operand_lts, lattices_ref res_lts) {
        for (auto [res_lt, operand_lt] : llvm::zip(res_lts, operand_lts))
            join(res_lt, *operand_lt);
        return mlir::success();
    }

    logical_result visit_branch(branch_iface, const_lattices_ref, lattices_ref) {
        return mlir::success();
    }

    logical_result visit_operation(
        mlir_operation *op, const_lattices_ref operand_lts, lattices_ref res_lts
    ) {
        return llvm::TypeSwitch< mlir::Operation *, logical_result >(op)
            .Case< pt::AddressOp,
                   pt::AllocOp,
                   pt::AssignOp,
                   pt::ConstantOp,
                   pt::CopyOp,
                   pt::DereferenceOp,
                   pt::NamedVarOp,
                   pt::UnknownPtrOp >
            ([&](auto &pt_op) {
                return visit_pt_op_impl(pt_op, operand_lts, res_lts);
            })
            .template Case< mlir::UnrealizedConversionCastOp >(
                    [&](auto &cast_op) {
                        return visit_unrealized_cast_impl(cast_op, operand_lts, res_lts);
            })
            .template Case< branch_iface >([&](auto &branch_op) {
                return visit_branch_impl(branch_op, operand_lts, res_lts);
            })
            .Default([&](auto &pt_op) {
                return visit_non_pt_op_impl(pt_op, operand_lts, res_lts);
            });
    }

    protected:
    symbol_table_collection tables;
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_func_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);
} // potato::analysis

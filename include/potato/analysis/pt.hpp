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
using mem_loc = std::pair< mlir_operation *, size_t >;

struct mem_loc_anchor : mlir::GenericLatticeAnchorBase< mem_loc_anchor, mem_loc > {
    using Base::Base;

    // allow constructing the anchor without a unique id
    mem_loc_anchor(mlir_operation *op) : Base(std::make_pair(op, 0)) {};

    mlir_operation *getAllocator() const { return getValue().first; }
    size_t getUniquer() const { return getValue().second; }

    void print(llvm::raw_ostream &os) const override;
};

struct pt_lattice_base : mlir::AnalysisState {
    pt_lattice_base(mlir_value value) : AnalysisState(value) {}

    change_result join(const pt_lattice_base &rhs);

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

template< typename derived_t >
struct pt_analysis : dfa {
    using pt_lattice = derived_t::lattice_t;
    using base = dfa;
    using const_lattices_ref = llvm::ArrayRef< const pt_lattice * >;
    using lattices_ref = llvm::ArrayRef< pt_lattice * >;

protected:

    derived_t &derived() const { return *static_cast< derived_t * >(this); }

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

    logical_result visit_call_operation_impl(
        mlir::CallOpInterface call,
        const_lattices_ref operand_lattices,
        lattices_ref result_lattices
    ) {
        return derived().visit_call_operation(call, operand_lattices, result_lattices);
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
           registerAnchorKind<mlir::dataflow::CFGEdge>();
           register_anchors_impl();
    }

    logical_result initialize(mlir_operation *root) override {
        for (mlir_region &region : root->getRegions()) {
            if (region.empty())
                continue;
            for (mlir_value argument : region.front().getArguments())
                set_to_entry_state_impl(get_lattice_element(argument));
        }
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
                    getProgramPointBefore(&block))->blockConentSubscribe(this);
                visit_block(&block);
            }
        }
    }

    constexpr bool add_deps() { return true; }

    logical_result visit_non_pt_op(mlir_operation *, const_lattices_ref, lattices_ref) {
        return mlir::success();
    }

    void visit_external_call(
        mlir::CallOpInterface call, const_lattices_ref arg_lattices, lattices_ref res_lattices
    ) {
        set_all_to_entry_states(res_lattices);
    }

    logical_result visit_call_operation(
        mlir::CallOpInterface call,
        const_lattices_ref operand_lattices,
        lattices_ref result_lattices
    ) {
        auto callable = dyn_cast_if_present< mlir::CallableOpInterface >(call.resolveCallable());
        if (!getSolverConfig().isInterprodcedural() ||
            (callable && !callable.getCallableRegion()))
        {
            visit_external_call_impl(call, operand_lattices, result_lattices);
            return mlir::success();
        }

        const auto predecessors = getOrCreateFor< mlir::dataflow::PredecessorState >(
            getProgramPointAfter(call), getProgramPointAfter(call)
        );

        // TODO: this sould simply check that we know all return sites from the function
        // Check if it's true and if we need to modify this in any way

        if (!predecessors->allPredecessorsKnown()) {
            set_all_to_entry_states(result_lattices);
            return mlir::success();
        }

        for (mlir_operation *predecessor : predecessors->getKnownPredecessors()) {
            for (auto &&[operand, res_lattice] :
                llvm::zip(predecessor->getOperands(), result_lattices))
            {
                join(res_lattice, *getLatticeElementFor(getProgramPointAfter(call), operand));
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
        // If not all callsites are known, conservatively mark all lattices as
        // having reached their pessimistic fixpoints.
        // TODO: This might be changed for out points-to analysis?
        if (!callsites->allePredecessorsKnown() || !getSolverConfig().isInterprocedural()) {
            set_all_to_entry_states(arg_lattices);
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
        set_all_to_entry_states(arg_lattices.take_front(first_index));
        set_all_to_entry_states(arg_lattices.drop_front(
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

    void set_all_to_entry_states(lattices_ref lattices) {
        for (auto lattice : lattices)
            set_to_entry_state_impl(lattice);
    }

    void join(pt_lattice *lhs, const pt_lattice &rhs) {
        propagateIfChanged(lhs, lhs->join(rhs));
    }

    void join_all(pt_lattice *lhs, const_lattices_ref rhs) {
        auto changed = change_result::NoChange;
        for (const auto &lattice : rhs)
            changed |= lhs->join(lattice);
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
            visit_call_operation_impl(call, operand_lattices, result_lattices);

        return visit_operation_impl(op, operand_lattices, result_lattices);
    }

    void visit_block(mlir_block *block) {
        if (block->getNumArguments() == 0)
            return;

        if (!getOrCreate<mlir::dataflow::Executable>(getProgramPointBefore(block))->isLive())
            return;
        mlir::SmallVector< pt_lattice * > arg_lattices;
        arg_lattices.reserver(block->getNumArguments());
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

            if (auto branch = dyn_cast< mlir::BranchOpInterface >(pred->getTerminator())) {
                auto operands = branch.getSuccessorOperands(it.getSuccessorIndex());
                for (auto [idx, lattice] : llvm::enumerate(arg_lattices)) {
                    if (auto operand = operands[idx]) {
                        join(lattice, *getLatticeElementFor(getProgramPointBefore(block)), operand);
                    } else {
                        // Conservatively consider internally produced arguments as entry points.
                        // TODO: is this necessary for us? when does this happen?
                        set_all_to_entry_states(lattice);
                    }
                }
            } else {
                return set_all_to_entry_states(arg_lattices);
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
                return setAllToEntryState(lattices);

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

    logical_result visit_pt_op(pt::AddressOp, const_lattices_ref operand_lts, lattices_ref res_lts);
    logical_result visit_pt_op(pt::AllocOp, const_lattices_ref operand_lts, lattices_ref res_lts);
    logical_result visit_pt_op(pt::AssignOp, const_lattices_ref operand_lts, lattices_ref res_lts);

    logical_result visit_pt_op(pt::ConstantOp, const_lattices_ref operand_lts, lattices_ref res_lts) {
        return mlir::success();
    }

    logical_result visit_pt_op(pt::CopyOp, const_lattices_ref operand_lts, lattices_ref res_lts) {
        for (const auto &operand_lat : operand_lts)
            join_all(res_lts, operand_lts);
        return mlir::success();
    }

    logical_result visit_pt_op(pt::DereferenceOp, const_lattices_ref operand_lts, lattices_ref res_lts);
    logical_result visit_pt_op(pt::NamedVarOp op, const_lattices_ref operand_lts, lattices_ref res_lts) {
        auto &init = op.getInit();
        if (!init.empty()) {
            auto *ret_op = &init.back().back();
            if (ret_op->hasTrait< mlir::OpTrait::ReturnLike >()) {
                for (auto ret_val : ret_op->getOperands()) {
                    // insert info into named var points-to state
                    assert(false);
                }
            }
        }
    }
    logical_result visit_pt_op(pt::UnknownPtrOp, const_lattices_ref operand_lts, lattices_ref res_lts);

    logical_result visit_operation(
        mlir_operation *op, const_lattices_ref operand_ltss, lattices_ref res_lts
    ) {
        llvm::TypeSwitch< mlir::Operation *, void >(op)
            .Case< pt::AddressOp,
                   pt::AllocOp,
                   pt::AssignOp,
                   pt::ConstantOp,
                   pt::CopyOp,
                   pt::DereferenceOp,
                   pt::NamedVarOp,
                   pt::UnknownPtrOp >
            ([&](auto &pt_op) {
                return visit_pt_op_impl(pt_op, operand_ltss, res_lts);
            })
            .template Case< mlir::UnrealizedConversionCastOp >(
                    [&](auto &cast_op) {}
            )
            .template Case< mlir::BranchOpInterface >([&](auto &branch_op) {})
            .Default([&](auto &pt_op) {
                return visit_non_pt_op_impl(pt_op, operand_ltss, res_lts);
            });
        return mlir::success();
    }
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_func_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);
} // potato::analysis

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/APSInt.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include "mlir/Interfaces/FunctionImplementation.h"
#include <mlir/Support/LLVM.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/util/common.hpp"

// TableGen generated stuff goes here:

#include "potato/dialect/PotatoDialect.cpp.inc"

using namespace potato::pt;

static mlir::ParseResult parseBranches(
        mlir::OpAsmParser &parser,
        mlir::SmallVector< mlir_block *, 2> &succs,
        mlir::DenseI32ArrayAttr &ranges,
        mlir::SmallVector< mlir::SmallVector< mlir::OpAsmParser::UnresolvedOperand > > &succ_ops,
        mlir::SmallVector< mlir::SmallVector< mlir::Type > > &succ_ops_types
) {
    mlir_block *succ;
    mlir::SmallVector< mlir::OpAsmParser::UnresolvedOperand > ops;
    mlir::SmallVector< mlir::Type > types;
    mlir::SmallVector< int32_t > sizes;
    do {
        if (mlir::failed(parser.parseSuccessor(succ)))
            return mlir::failure();
        if (mlir::succeeded(parser.parseOptionalLParen())) {
            if (mlir::failed(parser.parseOperandList(ops)))
               return  mlir::failure();
            if (ops.size() > 0) {
                if (mlir::failed(parser.parseColonTypeList(types)))
                    return mlir::failure();
            }
            if (mlir::failed(parser.parseRParen()))
                return mlir::failure();
        }
        if (ops.size() != types.size())
            return mlir::failure();
        sizes.push_back(ops.size());

        succs.push_back(succ);
        succ = nullptr;

        succ_ops.push_back(std::move(ops));
        ops.clear();

        succ_ops_types.push_back(std::move(types));
        types.clear();

    } while(mlir::succeeded(parser.parseOptionalComma()));

    ranges = mlir::DenseI32ArrayAttr::get(parser.getContext(), sizes);

    return mlir::success();
}

static void printBranches(
        mlir::OpAsmPrinter &printer,
        BranchOp op,
        mlir::SuccessorRange succs,
        mlir::ArrayRef< int32_t >,
        mlir::OperandRangeRange succ_operand_groups,
        mlir::TypeRangeRange succ_operand_types
) {
    auto sep = " ";
    for (size_t i = 0; i < succs.size(); i++) {
        printer << sep;
        printer.printSuccessorAndUseList(succs[i], succ_operand_groups[i]);
        sep = ", ";
    }
};

#define GET_OP_CLASSES
#include "potato/dialect/Potato.cpp.inc"

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
    return mlir::UnitAttr::get(this->getContext());
}

mlir::OpFoldResult ValuedConstantOp::fold(FoldAdaptor adaptor) {
    return adaptor.getValue();
}

mlir::OpFoldResult CopyOp::fold(FoldAdaptor) {
    mlir::OpFoldResult res{};
    for (auto operand : getOperands()) {
        auto def_op = operand.getDefiningOp();
        if (!def_op) {
            if (res)
                return {};
            res = operand;
            continue;
        }
        if (!(mlir::isa< pt::ConstantOp >(def_op))) {
            // Copy op is joining results of multiple non-constant operations,
            // conservatively bail out to not lose any information
            if (res)
                return {};
            res = operand;
        }
        if (mlir::isa< pt::UnknownPtrOp >(def_op)) {
            return operand;
        }
    }
    if (!res && this->getNumOperands() > 0) {
        res = getOperand(0);
    }
    if (auto operand = mlir::dyn_cast_if_present< mlir::Value >(res)) {
        operand.setLoc(mlir::FusedLoc::get(getContext(), {operand.getLoc(), this->getLoc()}));
    }
    return res;
}

logical_result AssignOp::canonicalize(AssignOp op, mlir::PatternRewriter &rewriter) {
    if (mlir::isa_and_present< pt::ConstantOp >(op.getRhs().getDefiningOp())) {
        rewriter.eraseOp(op);
        return mlir::success();
    }
    return mlir::failure();

}

mlir::SuccessorOperands BranchOp::getSuccessorOperands(unsigned idx) {
    assert(idx < getNumSuccessors() && "invalid successor index");
    return mlir::SuccessorOperands(getSuccOperandsMutable()[idx]);
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::StringAttr nameAttr;
  llvm::SmallVector< mlir::OpAsmParser::Argument > entryArgs;
  llvm::SmallVector< mlir::DictionaryAttr > resultAttrs;
  llvm::SmallVector< mlir_type > resultTypes;
  bool isVariadic;

  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      mlir::function_interface_impl::parseFunctionSignatureWithArguments(
          parser, /*allowVariadic=*/true, entryArgs, isVariadic, resultTypes,
          resultAttrs))
    return mlir::failure();

  llvm::SmallVector< mlir_type > argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  auto type = FunctionType::get(resultTypes.front(), argTypes, isVariadic);
  if (!type)
    return mlir::failure();
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      mlir::TypeAttr::get(type));


  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return mlir::failure();
  mlir::call_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, entryArgs, resultAttrs,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));

  auto *body = result.addRegion();
  auto parseResult = parser.parseOptionalRegion(*body, entryArgs);
  return mlir::failure(parseResult.has_value() && failed(*parseResult));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getName());

  FunctionType fnType = getFunctionType();
  llvm::SmallVector< mlir_type, 8 > argTypes;
  llvm::SmallVector< mlir_type, 1 > resTypes;
  argTypes.reserve(fnType.getNumParams());
  for (unsigned i = 0, e = fnType.getNumParams(); i < e; ++i)
    argTypes.push_back(fnType.getParamType(i));

  resTypes.push_back(fnType.getReturnType());

  mlir::function_interface_impl::printFunctionSignature(p, *this, argTypes,
                                                  isVarArg(), resTypes);


  mlir::function_interface_impl::printFunctionAttributes(
      p, *this,
      {getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName()});

  // Print the body if this is not an external function.
  auto &body = getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

logical_result CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  auto fn_attr = (*this)->getAttrOfType< mlir::FlatSymbolRefAttr >("callee");
  if (!fn_attr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fn_attr);
  if (!fn)
    return emitOpError() << "'" << fn_attr.getValue()
                         << "' does not reference a valid function";
  // TODO: verify args and returns
  return mlir::success();
}

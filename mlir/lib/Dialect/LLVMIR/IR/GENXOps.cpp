//===- GENXOps.cpp - MLIR GENX operations --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations in the GENX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/LLVMIR/GENXTypes.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// genx.matrix.load
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixLoadOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto resType = getResult().getType().cast<GENX::JointMatrixType>();
  if (getLayout() != resType.getMatrixLayout())
    return this->emitOpError("result layout must match layout attribute");

  return success();
}

//===----------------------------------------------------------------------===//
// genx.yield
//===----------------------------------------------------------------------===//

ParseResult GENX::YieldOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> opInfo;
  SmallVector<Type, 2> types;
  SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 parser.parseOptionalAttrDict(result.attributes) ||
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

void GENX::YieldOp::print(OpAsmPrinter &p) {
  if (getNumOperands() > 0)
    p << ' ' << getOperands();
  p.printOptionalAttrDict((*this)->getAttrs());
  if (getNumOperands() > 0)
    p << " : " << getOperandTypes();
}

LogicalResult GENX::YieldOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty())
    return emitOpError("expected single non-empty parent region");

  auto results = parentOp->getResults();
  auto operands = getOperands();

  if (!isa<GENX::MatrixMapOp>(parentOp))
    return emitOpError() << "only terminates genx.matrix.map regions";

  if (parentOp->getNumResults() != getNumOperands())
    return emitOpError() << "parent of yield must have same number of "
                            "results as the yield operands";

  for (auto it : llvm::zip(results, operands)) {
    if (std::get<0>(it).getType() != std::get<1>(it).getType())
      return emitOpError() << "types mismatch between yield op and its parent";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.map
//===----------------------------------------------------------------------===//

#if 0
ParseResult GENX::MatrixMapOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  // Parse <Scope>
  StringRef keyword;
  if (parser.parseLess())
    return failure();
  if (parser.parseKeyword(&keyword))
    return failure();
  if (!GENX::symbolizeEnum<Scope>(keyword))
    return failure();
  if (parser.parseGreater())
    return failure();

  // Parse operands
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOperandList(operands))
    return failure();

  // Parse lambda region
  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  llvm::errs() << "at line " << __LINE__ << "\n";

  return success();
}

void GENX::MatrixMapOp::print(OpAsmPrinter &p) {
  OperandRange operands(operand_begin(), operand_end());
  for (auto op : this->getOperands())
    p << '(' << op << ')';

  p.printOptionalAttrDict((*this)->getAttrs());

  p.increaseIndent();
  p.printNewline();
  p << "(";
  Block *mapper = getBody();
  llvm::interleaveComma(mapper->getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getMapper(), /*printEntryBlockArgs=*/false);
  p.decreaseIndent();
}
#endif

LogicalResult GENX::MatrixMapOp::verify() {
  // auto *bodyBlock = getBody();
  //  auto blockArgs = bodyBlock->getArguments();

  return success();
}

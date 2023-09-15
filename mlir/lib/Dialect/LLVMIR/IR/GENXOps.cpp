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
// genx.matrix.store
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixStoreOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto valType = getVal().getType().cast<GENX::JointMatrixType>();
  if (getLayout() != valType.getMatrixLayout())
    return this->emitOpError(
        "layout of value to store must match layout attribute");

  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.mad
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixMadOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto AType = getA().getType().cast<GENX::JointMatrixType>();
  auto BType = getB().getType().cast<GENX::JointMatrixType>();
  auto CType = getC().getType().cast<GENX::JointMatrixType>();
  auto resType = getResult().getType().cast<GENX::JointMatrixType>();

  if (CType != resType)
    return this->emitOpError("result and 3rd operand must have the same type");

  // Check the matrices dimensions match - A(M,K) * B(K,N) + C(M,N).
  if (AType.getNumRows() != resType.getNumRows() ||
      AType.getNumColumns() != BType.getNumRows() ||
      BType.getNumColumns() != resType.getNumColumns())
    return this->emitOpError("matrix sizes must match");

  Type AElemType = AType.getElementType();
  Type BElemType = BType.getElementType();
  Type CElemType = CType.getElementType();
  Type resElemType = resType.getElementType();

  // Check valid sizes for the matrixes dimensions which on XMX are:
  //   M <= 8, N == 16, K == 32 (if A's element type is integer)
  //   M <= 8, N == 16, K == 16 (if A's element type is floating-point)
  if (resType.getNumRows() > 8)
    return this->emitOpError("result matrix must have a max of 8 rows");
  if (resType.getNumColumns() != 16)
    return this->emitOpError("result matrix must have 16 columns");
  if (isa<IntegerType>(AElemType) && AType.getNumColumns() != 32)
    return this->emitOpError("1st operand matrix must have 32 columns");
  if (isa<FloatType>(AElemType) && AType.getNumColumns() != 16)
    return this->emitOpError("1st operand matrix must have 16 columns");

  // Check that element types match.
  if (AElemType != BElemType || resElemType != CElemType)
    return this->emitOpError("matrix element types must match");

  // Allowed matrices element types on XMX are:
  //   Matrices  |     A      |     B      |   C   |
  //   Elem Type | uint8/int8 | uint8/int8 | int32 |
  //             |    fp16    |    fp16    | fp32  |
  //             |    bf16    |    bf16    | fp32  |
  if (auto t = dyn_cast<IntegerType>(AElemType)) {
    if (t.getWidth() != 8)
      return this->emitOpError(
          "1st operand element type must have bit-width equal to 8");
    if (!isa<IntegerType>(CElemType) ||
        cast<IntegerType>(CElemType).getWidth() != 32)
      return this->emitOpError("3rd operand element type must be i32");
  } else if (auto t = dyn_cast<FloatType>(AElemType)) {
    if (!t.isF16() && !t.isBF16())
      return this->emitOpError("1st operand element type must be f16 or bf16");
    if (!isa<FloatType>(CElemType) ||
        cast<FloatType>(CElemType).getWidth() != 32)
      return this->emitOpError("3rd operand element type must be f32");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.init
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixInitOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto matType = getMat().getType().cast<GENX::JointMatrixType>();
  if (matType.getElementType() != getVal().getType())
    return this->emitOpError("initializer type must match matrix element type");

  return success();
}

//===----------------------------------------------------------------------===//
// genx.yield
//===----------------------------------------------------------------------===//

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
  Region &rgn = getBody();
  llvm::interleaveComma(rgn.getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(rgn, /*printEntryBlockArgs=*/false);
  p.decreaseIndent();
}

LogicalResult GENX::MatrixMapOp::verify() {
  // auto *bodyBlock = getBody();
  //  auto blockArgs = bodyBlock->getArguments();

  return success();
}

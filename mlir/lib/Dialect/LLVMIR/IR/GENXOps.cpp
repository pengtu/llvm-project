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
// genx.matrix.dpas
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixDPASOp::verify() {
  // TODO
  return success();
}

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
// genx.matrix.copy
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixCopyOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto resType = getResult().getType().cast<GENX::JointMatrixType>();
  auto srcType = getSrc().getType().cast<GENX::JointMatrixType>();

  if ((resType.getNumRows() != srcType.getNumRows()) ||
      (resType.getNumColumns() != srcType.getNumColumns()))
    return this->emitOpError("result shape must match source shape");

  if (resType.getMatrixLayout() != srcType.getMatrixLayout())
    return this->emitOpError("result layout must match source layout");

  return success();
}

//===----------------------------------------------------------------------===//
// genx.yield
//===----------------------------------------------------------------------===//

LogicalResult GENX::YieldOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (parentOp->getNumRegions() != 1 || parentOp->getRegion(0).empty())
    return emitOpError("expected single non-empty parent region");

  if (!isa<GENX::MatrixMapOp>(parentOp))
    return emitOpError() << "only terminates genx.matrix.map regions";

  if (parentOp->getNumResults() != getNumOperands())
    return emitOpError() << "parent of yield must have same number of "
                            "results as the yield operands";

  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.map
//===----------------------------------------------------------------------===//

void GENX::MatrixMapOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "mapped");
}

ParseResult GENX::MatrixMapOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  // Parse scope attribute.
  ScopeAttr scopeAttr;
  if (parser.parseCustomAttributeWithFallback(scopeAttr, Type{}, "scope",
                                              result.attributes))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  SmallVector<Type> inputTypes;
  SMLoc inputsOperandsLoc = parser.getCurrentLocation();

  // Parse input operands.
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperandList(inputsOperands) ||
      parser.parseColonTypeList(inputTypes) || parser.parseRParen())
    return failure();
  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands))
    return failure();

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse lambda.
  SmallVector<OpAsmParser::Argument> regionArgs;
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  // Parse result type.
  Type resType;
  if (parser.parseColon() || parser.parseCustomTypeWithFallback(resType))
    return failure();

  result.addTypes(resType);

  return success();
}

void GENX::MatrixMapOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printStrippedAttrOrType(getScopeAttr());
  p.printNewline();

  SmallVector<Value> inputs{this->getMat()};
  SmallVector<Type> inputTypes{this->getMat().getType()};
  for (Value input : this->getInputs()) {
    inputs.push_back(input);
    inputTypes.push_back(input.getType());
  }

  p << "ins(";
  p.printOperands(inputs);
  p << " : " << inputTypes << ")";

  SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("scope");
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  p.printNewline();
  p << "(";
  Block *body = getBody();
  llvm::interleaveComma(body->getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") ";

  p.printRegion(getMapper(), /*printEntryBlockArgs=*/false);

  p << " : ";
  auto resType = getRes().getType();
  if (auto validType = resType.dyn_cast<Type>())
    p.printStrippedAttrOrType(validType);
  else
    p << resType;
}

LogicalResult GENX::MatrixMapOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  SmallVector<Value> inputs{getMat()};
  for (Value input : getInputs())
    inputs.push_back(input);

  Block *bodyBlock = getBody();
  auto blockArgs = bodyBlock->getArguments();

  assert(inputs.size() == 2);

  // Checks that the arity of the `mapper` region is equal to the matrix
  // argument plus the number of variadic arguments.
  if (inputs.size() != blockArgs.size())
    return emitOpError() << "expects number of operands to match the arity of "
                            "mapper, but got: "
                         << inputs.size() << " and " << blockArgs.size();

  // The first parameters of the mapper should match the matrix element type.
  auto matType = cast<GENX::JointMatrixType>(inputs.front().getType());
  auto argTypes = bodyBlock->getArgumentTypes();
  if (matType.getElementType() != argTypes.front())
    return emitOpError() << "expected element type of input "
                         << matType.getElementType() << " to match bbArg type "
                         << argTypes.front();

  // The remaining parameters of the mapper should match the types of the
  // variadic arguments.
  for (size_t i = 1; i < argTypes.size(); ++i) {
    Type argType = argTypes[i];
    Type inType = inputs[i].getType();
    if (argType != inType)
      return emitOpError() << "expected type of input " << inType
                           << " to match bbArg type " << argType;
  }

  return success();
}

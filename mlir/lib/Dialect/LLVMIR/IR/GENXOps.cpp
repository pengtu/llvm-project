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

  if (AType.getNumRows() != resType.getNumRows() ||
      AType.getNumColumns() != BType.getNumRows() ||
      BType.getNumColumns() != resType.getNumColumns())
    return this->emitOpError("matrix sizes must match");

  if (AType.getElementType() != BType.getElementType() ||
      resType.getElementType() != CType.getElementType())
    return this->emitOpError("matrix element types must match");

  return success();
}

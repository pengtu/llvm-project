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

using namespace mlir;

//===----------------------------------------------------------------------===//
// genx.matrix.mad
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixMadOp::verify() {
  if (this->getC().getType() != this->getResult().getType())
    return this->emitOpError(
        "result and third operand must have the same type");
  auto typeA = this->getA().getType().cast<GENX::JointMatrixType>();
  auto typeB = this->getB().getType().cast<GENX::JointMatrixType>();
  auto typeC = this->getC().getType().cast<GENX::JointMatrixType>();
  auto typeR = this->getResult().getType().cast<GENX::JointMatrixType>();
  if (typeA.getNumRows() != typeR.getNumRows() ||
      typeA.getNumColumns() != typeB.getNumRows() ||
      typeB.getNumColumns() != typeR.getNumColumns())
    return this->emitOpError("matrix size must match");
  if (typeR.getScope() != typeA.getScope() ||
      typeR.getScope() != typeB.getScope() ||
      typeR.getScope() != typeC.getScope())
    return this->emitOpError("matrix scope must match");
  if (typeA.getElementType() != typeB.getElementType() ||
      typeR.getElementType() != typeC.getElementType())
    return this->emitOpError("matrix element type must match");
  return success();
}

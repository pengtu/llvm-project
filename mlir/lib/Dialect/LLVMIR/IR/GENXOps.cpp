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
<<<<<<< HEAD
#include "mlir/Dialect/LLVMIR/GENXTypes.h"
=======
>>>>>>> 5e9b906fb9be ([GENX][JointMatrix]: Add MatrixMad to the GENX dialect)

using namespace mlir;

//===----------------------------------------------------------------------===//
<<<<<<< HEAD
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
// genx.matrix.init
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixInitOp::verify() {
  // The scope attribute must be 'Subgroup' currently.
  if (getScope() != GENX::Scope::Subgroup)
    return this->emitOpError("scope attribute must have value 'Subgroup'");

  auto matType = getMat().getType().cast<GENX::JointMatrixType>();
  if (matType.getElementType() != getVal().getType())
    return this->emitOpError("initializer type must match matrix element type");

=======
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
>>>>>>> 5e9b906fb9be ([GENX][JointMatrix]: Add MatrixMad to the GENX dialect)
  return success();
}

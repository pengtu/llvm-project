//===- GENXTypes.h - MLIR GENX Types ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the types in the GENX dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_GENXTYPES_H_
#define MLIR_DIALECT_LLVMIR_GENXTYPES_H_

#include "mlir/Dialect/LLVMIR/GENXEnums.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace GENX {

namespace detail {
struct JointMatrixTypeStorage;
} // namespace detail

class GENXType : public Type {
public:
  using Type::Type;

  static bool classof(Type type);

  std::optional<int64_t> getSizeInBytes() const;
};

class CompositeType : public GENXType {
public:
  using GENXType::GENXType;

  static bool classof(Type type);

  unsigned getNumElements() const;

  Type getElementType(unsigned) const;

  std::optional<int64_t> getSizeInBytes() const;
};

class JointMatrixType : public Type::TypeBase<JointMatrixType, CompositeType,
                                              detail::JointMatrixTypeStorage> {
public:
  using Base::Base;

  static JointMatrixType get(Type elementType, Scope scope, unsigned rows,
                             unsigned columns, MatrixLayout matrixLayout);

  Type getElementType() const;

  Scope getScope() const;
  MatrixLayout getMatrixLayout() const;

  unsigned getRows() const;
  unsigned getColumns() const;
};

} // namespace GENX
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_GENXTYPES_H_

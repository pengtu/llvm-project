//===- GENXTypes.cpp - MLIR GENX Types ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types in the GENX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/GENXTypes.h"
#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/LLVMIR/GENXEnums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <iterator>

using namespace mlir;
using namespace mlir::GENX;

//===----------------------------------------------------------------------===//
// CompositeType
//===----------------------------------------------------------------------===//

bool CompositeType::classof(Type type) {
  return type.isa<GENX::JointMatrixType>();
}

unsigned CompositeType::getNumElements() const {
  if (isa<JointMatrixType>())
    llvm_unreachable(
        "invalid to query number of elements of GENX::JointMatrix type");

  llvm_unreachable("invalid composite type");
}

Type CompositeType::getElementType(unsigned index) const {
  return TypeSwitch<Type, Type>(*this)
      .Case<JointMatrixType>([](auto type) { return type.getElementType(); })
      .Default(
          [](Type) -> Type { llvm_unreachable("invalid composite type"); });
}

std::optional<int64_t> CompositeType::getSizeInBytes() const {
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// JointMatrixType
//===----------------------------------------------------------------------===//

struct GENX::detail::JointMatrixTypeStorage : public TypeStorage {
  using KeyTy = std::tuple<Type, unsigned, unsigned, MatrixLayout, Scope>;

  static JointMatrixTypeStorage *construct(TypeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<JointMatrixTypeStorage>())
        JointMatrixTypeStorage(key);
  }

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementType, rows, columns, matrixLayout, scope);
  }

  JointMatrixTypeStorage(const KeyTy &key)
      : elementType(std::get<0>(key)), rows(std::get<1>(key)),
        columns(std::get<2>(key)), scope(std::get<4>(key)),
        matrixLayout(std::get<3>(key)) {}

  Type elementType;
  unsigned rows;
  unsigned columns;
  Scope scope;
  MatrixLayout matrixLayout;
};

JointMatrixType JointMatrixType::get(Type elementType, Scope scope,
                                     unsigned rows, unsigned columns,
                                     MatrixLayout matrixLayout) {
  return Base::get(elementType.getContext(), elementType, rows, columns,
                   matrixLayout, scope);
}

Type JointMatrixType::getElementType() const { return getImpl()->elementType; }

Scope JointMatrixType::getScope() const { return getImpl()->scope; }

MatrixLayout JointMatrixType::getMatrixLayout() const {
  return getImpl()->matrixLayout;
}

unsigned JointMatrixType::getRows() const { return getImpl()->rows; }

unsigned JointMatrixType::getColumns() const { return getImpl()->columns; }

//===----------------------------------------------------------------------===//
// GENX Dialect
//===----------------------------------------------------------------------===//

void GENXDialect::registerTypes() { addTypes<JointMatrixType>(); }

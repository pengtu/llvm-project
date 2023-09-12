//===- GENXDialect.cpp - GENX IR Ops and Dialect registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the GENX IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The GENX dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/LLVMIR/GENXTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::GENX;

#include "mlir/Dialect/LLVMIR/GENXOpsDialect.cpp.inc"
#include "mlir/Dialect/LLVMIR/GENXOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/GENXOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// GENXDialect
//===----------------------------------------------------------------------===//

void GENXDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/GENXOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/GENXOpsAttributes.cpp.inc"
      >();

  // Support unknown operations because not all GENX operations are registered.
  allowUnknownOperations();
}

LogicalResult GENXDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attr) {
  // TODO: fill this in.
  return success();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

static Type parseAndVerifyType(GENXDialect const &dialect,
                               DialectAsmParser &parser) {
  Type type;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return Type();

  // Allow GENX dialect types
  if (&type.getDialect() == &dialect)
    return type;

  // Check other allowed types
  if (auto t = type.dyn_cast<FloatType>()) {
    if (type.isBF16()) {
      parser.emitError(typeLoc, "cannot use 'bf16' to compose GENX types");
      return Type();
    }
  } else if (auto t = type.dyn_cast<IntegerType>()) {
    if (!llvm::is_contained({1u, 8u, 16u, 32u, 64u}, t.getWidth())) {
      parser.emitError(typeLoc,
                       "only 1/8/16/32/64-bit integer type allowed but found ")
          << type;
      return Type();
    }
  } else if (auto t = type.dyn_cast<VectorType>()) {
    if (t.getRank() != 1) {
      parser.emitError(typeLoc, "only 1-D vector allowed but found ") << t;
      return Type();
    }
    if (t.getNumElements() > 4) {
      parser.emitError(
          typeLoc, "vector length has to be less than or equal to 4 but found ")
          << t.getNumElements();
      return Type();
    }
  } else {
    parser.emitError(typeLoc, "cannot use ")
        << type << " to compose GENX types";
    return Type();
  }

  return type;
}

/// Parses the next keyword in `parser` as an enumerator.
template <typename EnumClass, typename ParserType>
static ParseResult parseEnumKeywordAttr(EnumClass &value, ParserType &parser,
                                        StringRef attrName) {
  StringRef keyword;
  SmallVector<NamedAttribute, 1> attr;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&keyword))
    return failure();
  if (std::optional<EnumClass> attr = GENX::symbolizeEnum<EnumClass>(keyword)) {
    value = *attr;
    return success();
  }
  return parser.emitError(loc, "invalid ")
         << attrName << " attribute specification: " << keyword;
}

// joint-matrix-type ::= `!genx.jointmatrix` `<`rows `x` cols `x` element-type
//                                           `,` layout `,` scope`>`
static Type parseJointMatrixType(GENXDialect const &dialect,
                                 DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 2> dims;
  SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false))
    return Type();

  if (dims.size() != 2) {
    parser.emitError(countLoc, "expected rows and columns size");
    return Type();
  }

  auto elementTy = parseAndVerifyType(dialect, parser);
  if (!elementTy)
    return Type();
  MatrixLayout matrixLayout;
  if (parser.parseComma() ||
      parseEnumKeywordAttr(matrixLayout, parser, "matrixLayout <id>"))
    return Type();
  Scope scope;
  if (parser.parseComma() || parseEnumKeywordAttr(scope, parser, "scope <id>"))
    return Type();
  if (parser.parseGreater())
    return Type();
  return JointMatrixType::get(elementTy, scope, dims[0], dims[1], matrixLayout);
}

// Specialize this function to parse each of the parameters that define an
// ImageType. By default it assumes this is an enum type.
template <typename ValTy>
static std::optional<ValTy> parseAndVerify(GENXDialect const &dialect,
                                           DialectAsmParser &parser) {
  StringRef enumSpec;
  SMLoc enumLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&enumSpec)) {
    return std::nullopt;
  }

  auto val = GENX::symbolizeEnum<ValTy>(enumSpec);
  if (!val)
    parser.emitError(enumLoc, "unknown attribute: '") << enumSpec << "'";
  return val;
}

Type GENXDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "jointmatrix")
    return parseJointMatrixType(*this, parser);

  parser.emitError(parser.getNameLoc(), "unknown GENX-V type: ") << keyword;
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

static void print(JointMatrixType type, DialectAsmPrinter &os) {
  os << "jointmatrix<" << type.getRows() << "x" << type.getColumns() << "x";
  os << type.getElementType() << ", "
     << stringifyMatrixLayout(type.getMatrixLayout());
  os << ", " << stringifyScope(type.getScope()) << ">";
}

void GENXDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<JointMatrixType>([&](auto type) { print(type, os); })
      .Default([](Type) { llvm_unreachable("unhandled GENX type"); });
}

//===----------------------------------------------------------------------===//
// Attribute parsing
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseMemoryFenceFlags(OpAsmParser &parser,
                                               IntegerAttr &flagsAttr) {
  MemoryFenceFlagAttr memoryFenceFlagAttr;
  int flags = 0;
  do {
    if (parser.parseCustomAttributeWithFallback(memoryFenceFlagAttr))
      return failure();
    flags |= static_cast<int>(memoryFenceFlagAttr.getValue());
  } while (succeeded(parser.parseOptionalComma()));
  flagsAttr =
      IntegerAttr::get(IntegerType::get(parser.getContext(), 32), flags);
  return success();
}

Attribute GENXDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  if (type) {
    parser.emitError(parser.getNameLoc(), "unexpected type");
    return {};
  }

  StringRef attrKind;
  Attribute attr;
  OptionalParseResult result =
      generatedAttributeParser(parser, &attrKind, type, attr);
  if (result.has_value())
    return attr;

  parser.emitError(parser.getNameLoc(), "unknown GENX attribute kind: ")
      << attrKind;
  return {};
}

//===----------------------------------------------------------------------===//
// Attribute printing
//===----------------------------------------------------------------------===//

static void printMemoryFenceFlags(OpAsmPrinter &p, FenceOp op,
                                  IntegerAttr flags) {
  bool firstFlag = true;
  auto printFlag = [&](int flag) {
    assert((flag == 1) | (flag == 2) | (flag == 4) &&
           "Expecting valid memory fence flag");
    if (!firstFlag)
      p << ",";
    p.printStrippedAttrOrType(MemoryFenceFlagAttr::get(
        flags.getContext(), static_cast<MemoryFenceFlag>(flag)));
    firstFlag = false;
  };
  if (flags.getInt() & 1)
    printFlag(1);
  if (flags.getInt() & 2)
    printFlag(2);
  if (flags.getInt() & 4)
    printFlag(4);
}

void GENXDialect::printAttribute(Attribute attr,
                                 DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;

  llvm_unreachable("unhandled GENX attribute kind");
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/GENXOps.cpp.inc"

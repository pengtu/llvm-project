// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>, %c : !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>) {
  // expected-error @+1 {{'genx.matrix.mad' op matrix size must match}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor, Subgroup>, !genx.jointmatrix<8x8xi32, RowMajor, Subgroup> -> !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>
  llvm.return
}


// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>, %c : !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>) {
  // expected-error @+1 {{'genx.matrix.mad' op matrix size must match}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor, Subgroup>, !genx.jointmatrix<8x8xi32, RowMajor, Subgroup> -> !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor, Subgroup>, %b : !genx.jointmatrix<16x8xi32, RowMajor, Workgroup>, %c : !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>) {
  // expected-error @+1 {{'genx.matrix.mad' op matrix scope must match}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor, Subgroup>, !genx.jointmatrix<16x8xi32, RowMajor, Workgroup> -> !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>
  llvm.return  
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xf32, RowMajor, Subgroup>, %b : !genx.jointmatrix<16x8xi32, RowMajor, Subgroup>, %c : !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>) {
  // expected-error @+1 {{matrix element type must match}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xf32, RowMajor, Subgroup>, !genx.jointmatrix<16x8xi32, RowMajor, Subgroup> -> !genx.jointmatrix<8x8xi32, RowMajor, Subgroup>
  llvm.return
}

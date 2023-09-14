// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @joint_matrix_load(%ptr : !llvm.ptr<i32>, %stride : index) {
  // expected-error @+1 {{'genx.matrix.load' op scope attribute must have value 'Subgroup'}}
  %0 = genx.matrix.load <Workgroup> <RowMajor> %ptr, %stride {memory_access = #genx.memory_access<Volatile>} : (!llvm.ptr<i32>, index) -> !genx.jointmatrix<8x16xi32, RowMajor>
  llvm.return
}

// -----

func.func @joint_matrix_load(%ptr : !llvm.ptr<i32>, %stride : index) {
  // expected-error @+1 {{'genx.matrix.load' op result layout must match layout attribute}}
  %1 = genx.matrix.load <Subgroup> <RowMajor> %ptr, %stride {memory_access = #genx.memory_access<Volatile>} : (!llvm.ptr<i32>, index) -> !genx.jointmatrix<8x16xi32, ColumnMajor>
  llvm.return
}


// -----

func.func @joint_matrix_copy(%src : !genx.jointmatrix<8x16xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.copy' op scope attribute must have value 'Subgroup'}}
  %0 = genx.matrix.copy <Workgroup> %src : (!genx.jointmatrix<8x16xi32, RowMajor>) -> !genx.jointmatrix<8x16xi32, RowMajor>
  llvm.return
}

// -----

func.func @joint_matrix_copy(%src : !genx.jointmatrix<8x16xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.copy' op result shape must match source shape}}
  %0 = genx.matrix.copy <Subgroup> %src : (!genx.jointmatrix<8x16xi32, RowMajor>) -> !genx.jointmatrix<16x8xi32, RowMajor>
  llvm.return
}

// -----

func.func @joint_matrix_copy(%src : !genx.jointmatrix<8x16xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.copy' op result layout must match source layout}}
  %0 = genx.matrix.copy <Subgroup> %src : (!genx.jointmatrix<8x16xi32, RowMajor>) -> !genx.jointmatrix<8x16xf32, ColumnMajor>
  llvm.return
}

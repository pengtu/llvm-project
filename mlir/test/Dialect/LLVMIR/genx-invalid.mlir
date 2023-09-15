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

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor>, %b : !genx.jointmatrix<8x8xi32, RowMajor>, %c : !genx.jointmatrix<8x8xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op scope attribute must have value 'Subgroup'}}
  %r = genx.matrix.mad <Workgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor> -> !genx.jointmatrix<8x8xi32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xi32, RowMajor>, %b : !genx.jointmatrix<8x8xi32, RowMajor>, %c : !genx.jointmatrix<8x8xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op matrix sizes must match}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor> -> !genx.jointmatrix<8x8xi32, RowMajor>
  llvm.return    
}

// -----

func.func @joint_matrix_mad(%a : !genx.jointmatrix<8x16xf32, RowMajor>, %b : !genx.jointmatrix<16x8xi32, RowMajor>, %c : !genx.jointmatrix<8x8xi32, RowMajor>) {
  // expected-error @+1 {{'genx.matrix.mad' op matrix element types must match}}
  %r = genx.matrix.mad <Subgroup> %a, %b, %c : !genx.jointmatrix<8x16xf32, RowMajor>, !genx.jointmatrix<16x8xi32, RowMajor>, !genx.jointmatrix<8x8xi32, RowMajor> -> !genx.jointmatrix<8x8xi32, RowMajor>
  llvm.return
}



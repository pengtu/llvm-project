// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

llvm.func @genx_special_regs() -> i64 {
  // CHECK-LABEL: genx_special_regs
  // CHECK: call i64 @_Z12get_local_idj(i32 0)
  %1 = genx.workitem.id.x : i64
  // CHECK: call i64 @_Z12get_local_idj(i32 1)
  %2 = genx.workitem.id.y : i64
  // CHECK: call i64 @_Z12get_local_idj(i32 2)
  %3 = genx.workitem.id.z : i64
  // CHECK: call i64 @_Z12get_group_idj(i32 0)
  %4 = genx.workgroup.id.x : i64
  // CHECK: call i64 @_Z12get_group_idj(i32 1)
  %5 = genx.workgroup.id.y : i64
  // CHECK: call i64 @_Z12get_group_idj(i32 2)
  %6 = genx.workgroup.id.z : i64
  // CHECK: call i64 @_Z12get_local_sizej(i32 0)
  %7 = genx.workgroup.dim.x : i64
  // CHECK: call i64 @_Z12get_local_sizej(i32 1)
  %8 = genx.workgroup.dim.y : i64
  // CHECK: call i64 @_Z12get_local_sizej(i32 2)
  %9 = genx.workgroup.dim.z : i64
  // CHECK: call i64 @_Z12get_global_sizej(i32 0)
  %10 = genx.grid.dim.x : i64
  // CHECK: call i64 @_Z12get_global_sizej(i32 1)
  %11 = genx.grid.dim.y : i64
  // CHECK: call i64 @_Z12get_global_sizej(i32 2)
  %12 = genx.grid.dim.z : i64

  llvm.return %1 : i64
}

llvm.func @genx.barrier() {
  // CHECK: call void @_Z7barrierj(i32 3)
  genx.barrier
  llvm.return
}

llvm.func @genx.sub_group_shuffle() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = call i32 @_Z21sub_group_shuffle_xorij(i32 0, i32 0)
  %1 = genx.sub_group_shuffle XOR %0, %0 : i32 -> i32
  // CHECK: %2 = call i32 @_Z20sub_group_shuffle_upij(i32 0, i32 0)
  %2 = genx.sub_group_shuffle UP %0, %0 : i32 -> i32
  // CHECK: %3 = call i32 @_Z22sub_group_shuffle_downij(i32 0, i32 0)
  %3 = genx.sub_group_shuffle DOWN %0, %0 : i32 -> i32
  %4 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %4 = call i8 @_Z21sub_group_shuffle_xorcj(i8 0, i32 0)
  %5 = genx.sub_group_shuffle XOR %4, %0 : i8 -> i8
  %6 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %5 = call i16 @_Z21sub_group_shuffle_xorsj(i16 0, i32 0)
  %7 = genx.sub_group_shuffle XOR %6, %0 : i16 -> i16
  %8 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %6 = call i64 @_Z21sub_group_shuffle_xorlj(i64 0, i32 0)
  %9 = genx.sub_group_shuffle XOR %8, %0 : i64 -> i64
  %10 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %7 = call half @_Z21sub_group_shuffle_xorDhj(half 0xH0000, i32 0)
  %11 = genx.sub_group_shuffle XOR %10, %0 : f16 -> f16
  %12 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %8 = call float @_Z21sub_group_shuffle_xorfj(float 0.000000e+00, i32 0)
  %13 = genx.sub_group_shuffle XOR %12, %0 : f32 -> f32
  %14 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %9 = call double @_Z21sub_group_shuffle_xordj(double 0.000000e+00, i32 0)
  %15 = genx.sub_group_shuffle XOR %14, %0 : f64 -> f64
  llvm.return
}

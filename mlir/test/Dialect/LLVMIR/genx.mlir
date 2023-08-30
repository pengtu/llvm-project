// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

func.func @genx_special_regs() -> i32 {
  // CHECK-LABEL: genx_special_regs
  // CHECK: genx.workitem.id.x : i32
  %0 = genx.workitem.id.x : i32
  // CHECK: genx.workitem.id.y : i32
  %1 = genx.workitem.id.y : i32
  // CHECK: genx.workitem.id.z : i32
  %2 = genx.workitem.id.z : i32
  // CHECK: genx.workgroup.id.x : i32
  %3 = genx.workgroup.id.x : i32
  // CHECK: genx.workgroup.id.y : i32
  %4 = genx.workgroup.id.y : i32
  // CHECK: genx.workgroup.id.z : i32
  %5 = genx.workgroup.id.z : i32
  // CHECK: genx.workgroup.dim.x : i32
  %6 = genx.workgroup.dim.x : i32
  // CHECK: genx.workgroup.dim.y : i32
  %7 = genx.workgroup.dim.y : i32
  // CHECK: genx.workgroup.dim.z : i32
  %8 = genx.workgroup.dim.z : i32
  // CHECK: genx.grid.dim.x : i32
  %9 = genx.grid.dim.x : i32
  // CHECK: genx.grid.dim.y : i32
  %10 = genx.grid.dim.y : i32
  // CHECK: genx.grid.dim.z : i32
  %11 = genx.grid.dim.z : i32
  llvm.return %0 : i32
}

func.func @genx.barrier() {
  // CHECK: genx.barrier
  genx.barrier
  llvm.return
}

func.func @genx.sub_group_shuffle() {
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = genx.sub_group_shuffle XOR %0, %0 : i32 -> i32
  %1 = genx.sub_group_shuffle XOR %0, %0 : i32 -> i32
  // CHECK: %2 = genx.sub_group_shuffle UP %0, %0 : i32 -> i32
  %2 = genx.sub_group_shuffle UP %0, %0 : i32 -> i32
  // CHECK: %3 = genx.sub_group_shuffle DOWN %0, %0 : i32 -> i32
  %3 = genx.sub_group_shuffle DOWN %0, %0 : i32 -> i32
  %4 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %5 = genx.sub_group_shuffle XOR %4, %0 : i8 -> i8
  %5 = genx.sub_group_shuffle XOR %4, %0 : i8 -> i8
  %6 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %7 = genx.sub_group_shuffle XOR %6, %0 : i16 -> i16
  %7 = genx.sub_group_shuffle XOR %6, %0 : i16 -> i16
  %8 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %9 = genx.sub_group_shuffle XOR %8, %0 : i64 -> i64
  %9 = genx.sub_group_shuffle XOR %8, %0 : i64 -> i64
  %10 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %11 = genx.sub_group_shuffle XOR %10, %0 : f16 -> f16
  %11 = genx.sub_group_shuffle XOR %10, %0 : f16 -> f16
  %12 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %13 = genx.sub_group_shuffle XOR %12, %0 : f32 -> f32
  %13 = genx.sub_group_shuffle XOR %12, %0 : f32 -> f32
  %14 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %15 = genx.sub_group_shuffle XOR %14, %0 : f64 -> f64
  %15 = genx.sub_group_shuffle XOR %14, %0 : f64 -> f64
  llvm.return
}

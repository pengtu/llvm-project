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
  // CHECK-LABEL: genx.barrier
  // CHECK: genx.barrier
  genx.barrier
  llvm.return
}

func.func @genx.atomic_work_item_fence() {
  // CHECK-LABEL: genx.atomic_work_item_fence
  // CHECK: genx.atomic_work_item_fence < LOCAL_MEM_FENCE >,  Relaxed,  work_item
  genx.atomic_work_item_fence <LOCAL_MEM_FENCE>, Relaxed, work_item
  // CHECK: genx.atomic_work_item_fence < GLOBAL_MEM_FENCE >,  Acquire,  work_group
  genx.atomic_work_item_fence <GLOBAL_MEM_FENCE>, Acquire, work_group
  // CHECK: genx.atomic_work_item_fence < IMAGE_MEM_FENCE >,  Release,  device
  genx.atomic_work_item_fence <IMAGE_MEM_FENCE>, Release, device
  // CHECK: genx.atomic_work_item_fence < LOCAL_MEM_FENCE >,  AcquireRelease,  all_svm_devices
  genx.atomic_work_item_fence <LOCAL_MEM_FENCE>, AcquireRelease, all_svm_devices
  // CHECK: genx.atomic_work_item_fence < GLOBAL_MEM_FENCE >,  SequentiallyConsistent,  sub_group
  genx.atomic_work_item_fence <GLOBAL_MEM_FENCE>, SequentiallyConsistent, sub_group
  // CHECK: genx.atomic_work_item_fence < LOCAL_MEM_FENCE, IMAGE_MEM_FENCE >,  Acquire,  sub_group
  genx.atomic_work_item_fence <LOCAL_MEM_FENCE, IMAGE_MEM_FENCE>, Acquire, sub_group
  llvm.return
}

func.func @genx.sub_group_shuffle() {
  // CHECK-LABEL: genx.sub_group_shuffle
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = genx.sub_group_shuffle XOR %0, %0 : i32 -> i32
  %1 = genx.sub_group_shuffle XOR %0, %0 : i32 -> i32
  // CHECK: %2 = genx.sub_group_shuffle UP %0, %0 : i32 -> i32
  %2 = genx.sub_group_shuffle UP %0, %0 : i32 -> i32
  // CHECK: %3 = genx.sub_group_shuffle DOWN %0, %0 : i32 -> i32
  %3 = genx.sub_group_shuffle DOWN %0, %0 : i32 -> i32
  // CHECK: %4 = genx.sub_group_shuffle IDX %0, %0 : i32 -> i32
  %4 = genx.sub_group_shuffle IDX %0, %0 : i32 -> i32
  %5 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %6 = genx.sub_group_shuffle XOR %5, %0 : i8 -> i8
  %6 = genx.sub_group_shuffle XOR %5, %0 : i8 -> i8
  %7 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %8 = genx.sub_group_shuffle XOR %7, %0 : i16 -> i16
  %8 = genx.sub_group_shuffle XOR %7, %0 : i16 -> i16
  %9 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %10 = genx.sub_group_shuffle XOR %9, %0 : i64 -> i64
  %10 = genx.sub_group_shuffle XOR %9, %0 : i64 -> i64
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %12 = genx.sub_group_shuffle XOR %11, %0 : f16 -> f16
  %12 = genx.sub_group_shuffle XOR %11, %0 : f16 -> f16
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %14 = genx.sub_group_shuffle XOR %13, %0 : f32 -> f32
  %14 = genx.sub_group_shuffle XOR %13, %0 : f32 -> f32
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %16 = genx.sub_group_shuffle XOR %15, %0 : f64 -> f64
  %16 = genx.sub_group_shuffle XOR %15, %0 : f64 -> f64
  llvm.return
}

llvm.func @genx.atomic.cmpxchg.i32(%ptr : !llvm.ptr<i32>, %cmp : i32, %val : i32)  {
  // CHECK-LABEL: genx.atomic.cmpxchg.i32
  // CHECK: %0 = genx.atomic.cmpxchg %arg0, %arg1, %arg2 : (!llvm.ptr<i32>, i32, i32) -> i32
  %0 = genx.atomic.cmpxchg %ptr, %cmp, %val : (!llvm.ptr<i32>, i32, i32) -> i32
  llvm.return
}

llvm.func @genx.atomic.rmw.i32(%ptr : !llvm.ptr<i32>, %val : i32) {
  // CHECK-LABEL: genx.atomic.rmw.i32
  // CHECK: genx.atomic.rmw AND %arg0, %arg1 : (!llvm.ptr<i32>, i32) -> i32
  %0 = genx.atomic.rmw AND %ptr, %val : (!llvm.ptr<i32>, i32) -> i32
  // CHECK: genx.atomic.rmw OR %arg0, %arg1 : (!llvm.ptr<i32>, i32) -> i32  
  %1 = genx.atomic.rmw OR %ptr, %val : (!llvm.ptr<i32>, i32) -> i32
  // CHECK: genx.atomic.rmw XOR %arg0, %arg1 : (!llvm.ptr<i32>, i32) -> i32  
  %2 = genx.atomic.rmw XOR %ptr, %val : (!llvm.ptr<i32>, i32) -> i32
  // CHECK: genx.atomic.rmw ADD %arg0, %arg1 : (!llvm.ptr<i32>, i32) -> i32  
  %3 = genx.atomic.rmw ADD %ptr, %val : (!llvm.ptr<i32>, i32) -> i32
  // CHECK: genx.atomic.rmw MIN %arg0, %arg1 : (!llvm.ptr<i32>, i32) -> i32  
  %4 = genx.atomic.rmw MIN %ptr, %val : (!llvm.ptr<i32>, i32) -> i32
  // CHECK: genx.atomic.rmw MAX %arg0, %arg1 : (!llvm.ptr<i32>, i32) -> i32  
  %5 = genx.atomic.rmw MAX %ptr, %val : (!llvm.ptr<i32>, i32) -> i32
  // CHECK: genx.atomic.rmw MAX %arg0, %arg1 : (!llvm.ptr<i32>, i32) -> i32  
  %6 = genx.atomic.rmw MAX %ptr, %val : (!llvm.ptr<i32>, i32) -> i32
  // CHECK: genx.atomic.rmw XCHG %arg0, %arg1 : (!llvm.ptr<i32>, i32) -> i32  
  %7 = genx.atomic.rmw XCHG %ptr, %val : (!llvm.ptr<i32>, i32) -> i32
  llvm.return  
}

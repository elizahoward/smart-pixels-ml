; ModuleID = '/home/dabadjiev/smartpixels_ml_dsabadjiev/smart-pixels-ml/hlsTmpModel2/myproject_prj/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>" = type { %"struct.ap_fixed_base<16, 6, true, AP_TRN, AP_WRAP, 0>" }
%"struct.ap_fixed_base<16, 6, true, AP_TRN, AP_WRAP, 0>" = type { %"struct.ssdm_int<16, true>" }
%"struct.ssdm_int<16, true>" = type { i16 }
%"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>" = type { %"struct.ap_fixed_base<8, 1, true, AP_RND_CONV, AP_SAT, 0>" }
%"struct.ap_fixed_base<8, 1, true, AP_RND_CONV, AP_SAT, 0>" = type { %"class.std::ios_base::Init" }
%"class.std::ios_base::Init" = type { i8 }

; Function Attrs: inaccessiblemem_or_argmemonly noinline willreturn
define void @apatb_myproject_ir(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="21" %x_profile, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %z_global, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="13" %y_profile, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %y_local, %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"* noalias nocapture nonnull "fpga.decayed.dim.hint"="1" "partition" %layer25_out) local_unnamed_addr #0 {
entry:
  %x_profile_copy7 = alloca i336, align 512
  %z_global_copy8 = alloca i16, align 512
  %y_profile_copy9 = alloca i208, align 512
  %y_local_copy10 = alloca i16, align 512
  %layer25_out_copy6 = alloca i8, align 512
  %0 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %x_profile to [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %1 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %z_global to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %2 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %y_profile to [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %3 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %y_local to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %4 = bitcast %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"* %layer25_out to [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]*
  call void @copy_in([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %0, i336* nonnull align 512 %x_profile_copy7, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %1, i16* nonnull align 512 %z_global_copy8, [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %2, i208* nonnull align 512 %y_profile_copy9, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %3, i16* nonnull align 512 %y_local_copy10, [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* nonnull %4, i8* nonnull align 512 %layer25_out_copy6)
  call void @apatb_myproject_hw(i336* %x_profile_copy7, i16* %z_global_copy8, i208* %y_profile_copy9, i16* %y_local_copy10, i8* %layer25_out_copy6)
  call void @copy_back([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i336* %x_profile_copy7, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %1, i16* %z_global_copy8, [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2, i208* %y_profile_copy9, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %3, i16* %y_local_copy10, [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %4, i8* %layer25_out_copy6)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"(i8* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"], [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = load i8, i8* %src.addr.0.0.05, align 1
  store i8 %1, i8* %dst, align 1
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"(i8* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"(i8* %dst, [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* nonnull %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>.134"([1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i8* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr.0.0.06 = getelementptr [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"], [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = load i8, i8* %src, align 1
  store i8 %1, i8* %dst.addr.0.0.06, align 1
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>.131"([1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i8* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>.134"([1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* nonnull %dst, i8* %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a21struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i336* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, i64 %dst_shift, [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = mul i64 16, %for.loop.idx2
  %2 = add i64 %dst_shift, %1
  %3 = load i16, i16* %src.addr.0.0.05, align 2
  %4 = load i336, i336* %dst, align 64
  %5 = zext i64 %2 to i336
  %6 = shl i336 65535, %5
  %7 = zext i16 %3 to i336
  %8 = shl i336 %7, %5
  %thr.xor1 = xor i336 %6, -1
  %thr.and2 = and i336 %4, %thr.xor1
  %thr.or3 = or i336 %thr.and2, %8
  store i336 %thr.or3, i336* %dst, align 64
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a21struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i336* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #2 {
entry:
  %0 = icmp eq [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a21struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i336* %dst, i64 0, [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 21)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.144"(i16* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, i64 %dst_shift, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  %1 = trunc i64 %dst_shift to i16
  %2 = shl i16 -1, %1
  %3 = xor i16 %2, -1
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %4 = load i16, i16* %src.addr.0.0.05, align 2
  %5 = load i16, i16* %dst, align 2
  %6 = shl i16 %4, %1
  %7 = and i16 %5, %3
  %8 = or i16 %7, %6
  store i16 %8, i16* %dst, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.141"(i16* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.144"(i16* %dst, i64 0, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a13struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i208* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, i64 %dst_shift, [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = mul i64 16, %for.loop.idx2
  %2 = add i64 %dst_shift, %1
  %3 = load i16, i16* %src.addr.0.0.05, align 2
  %4 = load i208, i208* %dst, align 32
  %5 = zext i64 %2 to i208
  %6 = shl i208 65535, %5
  %7 = zext i16 %3 to i208
  %8 = shl i208 %7, %5
  %thr.xor1 = xor i208 %6, -1
  %thr.and2 = and i208 %4, %thr.xor1
  %thr.or3 = or i208 %thr.and2, %8
  store i208 %thr.or3, i208* %dst, align 32
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a13struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i208* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #2 {
entry:
  %0 = icmp eq [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a13struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i208* %dst, i64 0, [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 13)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_in([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="0" "unpacked"="0", i336* noalias nocapture align 512 "orig.arg.no"="1" "unpacked"="1.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture align 512 "orig.arg.no"="3" "unpacked"="3.0", [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="4" "unpacked"="4", i208* noalias nocapture align 512 "orig.arg.no"="5" "unpacked"="5.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="6" "unpacked"="6", i16* noalias nocapture align 512 "orig.arg.no"="7" "unpacked"="7.0", [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* noalias readonly "orig.arg.no"="8" "unpacked"="8", i8* noalias nocapture align 512 "orig.arg.no"="9" "unpacked"="9.0") #3 {
entry:
  call void @"onebyonecpy_hls.p0a21struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i336* align 512 %1, [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.141"(i16* align 512 %3, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2)
  call void @"onebyonecpy_hls.p0a13struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i208* align 512 %5, [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %4)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.141"(i16* align 512 %7, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"(i8* align 512 %9, [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %8)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a21struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.177"([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i336* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 %src_shift, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %1 = mul i64 16, %for.loop.idx2
  %2 = add i64 %src_shift, %1
  %dst.addr.0.0.06 = getelementptr [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %3 = load i336, i336* %src, align 64
  %4 = zext i64 %2 to i336
  %5 = lshr i336 %3, %4
  %6 = trunc i336 %5 to i16
  store i16 %6, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a21struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.174"([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i336* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #2 {
entry:
  %0 = icmp eq [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a21struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.177"([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i336* %src, i64 0, i64 21)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.151"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i16* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 %src_shift, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  %1 = trunc i64 %src_shift to i16
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr.0.0.06 = getelementptr [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %2 = load i16, i16* %src, align 2
  %3 = lshr i16 %2, %1
  store i16 %3, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i16* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.151"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i16* %src, i64 0, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a13struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.160"([13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i208* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 %src_shift, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %1 = mul i64 16, %for.loop.idx2
  %2 = add i64 %src_shift, %1
  %dst.addr.0.0.06 = getelementptr [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %3 = load i208, i208* %src, align 32
  %4 = zext i64 %2 to i208
  %5 = lshr i208 %3, %4
  %6 = trunc i208 %5 to i16
  store i16 %6, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a13struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.157"([13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i208* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #2 {
entry:
  %0 = icmp eq [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a13struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.160"([13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i208* %src, i64 0, i64 13)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_out([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i336* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0", [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="4" "unpacked"="4", i208* noalias nocapture readonly align 512 "orig.arg.no"="5" "unpacked"="5.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="6" "unpacked"="6", i16* noalias nocapture readonly align 512 "orig.arg.no"="7" "unpacked"="7.0", [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* noalias "orig.arg.no"="8" "unpacked"="8", i8* noalias nocapture readonly align 512 "orig.arg.no"="9" "unpacked"="9.0") #4 {
entry:
  call void @"onebyonecpy_hls.p0a21struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.174"([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i336* align 512 %1)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2, i16* align 512 %3)
  call void @"onebyonecpy_hls.p0a13struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.157"([13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %4, i208* align 512 %5)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6, i16* align 512 %7)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>.131"([1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %8, i8* align 512 %9)
  ret void
}

declare void @apatb_myproject_hw(i336*, i16*, i208*, i16*, i8*)

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_back([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i336* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0", [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="4" "unpacked"="4", i208* noalias nocapture readonly align 512 "orig.arg.no"="5" "unpacked"="5.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="6" "unpacked"="6", i16* noalias nocapture readonly align 512 "orig.arg.no"="7" "unpacked"="7.0", [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* noalias "orig.arg.no"="8" "unpacked"="8", i8* noalias nocapture readonly align 512 "orig.arg.no"="9" "unpacked"="9.0") #4 {
entry:
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>.131"([1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %8, i8* align 512 %9)
  ret void
}

define void @myproject_hw_stub_wrapper(i336*, i16*, i208*, i16*, i8*) #5 {
entry:
  %5 = alloca [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %6 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %7 = alloca [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %8 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %9 = alloca [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]
  call void @copy_out([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %5, i336* %0, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6, i16* %1, [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %7, i208* %2, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %8, i16* %3, [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %9, i8* %4)
  %10 = bitcast [21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %5 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %11 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %12 = bitcast [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %7 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %13 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %8 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %14 = bitcast [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %9 to %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"*
  call void @myproject_hw_stub(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %10, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %11, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %12, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %13, %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"* %14)
  call void @copy_in([21 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %5, i336* %0, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6, i16* %1, [13 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %7, i208* %2, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %8, i16* %3, [1 x %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"]* %9, i8* %4)
  ret void
}

declare void @myproject_hw_stub(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<8, 1, AP_RND_CONV, AP_SAT, 0>"* noalias nocapture nonnull)

attributes #0 = { inaccessiblemem_or_argmemonly noinline willreturn "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="arraycpy_hls" }
attributes #2 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #3 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyin" }
attributes #4 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyout" }
attributes #5 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}
!datalayout.transforms.on.top = !{!5}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
!5 = !{!6, !8, !10}
!6 = !{!7}
!7 = !{!"4.0", [1 x i8]* null}
!8 = !{!9}
!9 = !{!"array_partition", !"type=Complete", !"dim=1"}
!10 = !{!11}
!11 = !{!"4.0", i8* null}

// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Metal indirect command buffers (macOS): the MLX pass's program encoded once a shape and replayed, each instruction
//! a dispatch the GPU schedules itself. A context holds its buffers (its own, or aliases of memory MLX allocated:
//! page-aligned, never copied), a library compiled from source with MLX's math (safe, so the bits are the step's),
//! pipelines an indirect command buffer can run, and the buffers of commands. Handles are indices into tables one
//! mutex holds; every failure is a code (and, where it has one, Metal's message), never a fault.

use std::ffi::{c_char, c_void, CStr};
use std::ptr::NonNull;
use std::sync::Mutex;

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::{NSObjectProtocol, ProtocolObject};
use objc2::sel;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineDescriptor,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLDispatchType,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLIndirectComputeCommand, MTLLibrary, MTLMathMode, MTLPipelineOption, MTLResource,
    MTLResourceOptions, MTLResourceUsage, MTLSize,
};

use crate::codes::*;

struct Ctx {
    dev: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    lib: Option<Retained<ProtocolObject<dyn MTLLibrary>>>,
    // a freed buffer's or command buffer's slot left empty, so no index ever names another
    bufs: Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    psos: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    icbs: Vec<Option<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>>,
}

// Metal's devices, queues, buffers, pipelines and indirect command buffers may be used from any thread (Apple's Metal
// threading notes); a context is only reached through `CTXS`'s lock, so no two threads encode into one at once
unsafe impl Send for Ctx {}

static CTXS: Mutex<Vec<Option<Ctx>>> = Mutex::new(Vec::new());

fn with<R>(h: i64, f: impl FnOnce(&mut Ctx) -> Result<R, i32>) -> Result<R, i32> {
    let mut all = CTXS.lock().map_err(|_| ERR_PANIC)?;
    let c = usize::try_from(h)
        .ok()
        .and_then(|i| all.get_mut(i))
        .and_then(|c| c.as_mut())
        .ok_or(ERR_DOMAIN)?;
    autoreleasepool(|_| f(c))
}

fn idx<T>(v: &[T], i: i64) -> Result<&T, i32> {
    usize::try_from(i)
        .ok()
        .and_then(|i| v.get(i))
        .ok_or(ERR_DOMAIN)
}

/// slot `i`'s object, `ERR_DOMAIN` for one out of range or freed
fn slot<T>(v: &[Option<T>], i: i64) -> Result<&T, i32> {
    idx(v, i)?.as_ref().ok_or(ERR_DOMAIN)
}

/// slot `i` emptied, `ERR_DOMAIN` for one out of range or freed already
fn free<T>(v: &mut [Option<T>], i: i64) -> Result<i32, i32> {
    let s = usize::try_from(i)
        .ok()
        .and_then(|i| v.get_mut(i))
        .ok_or(ERR_DOMAIN)?;
    s.take().ok_or(ERR_DOMAIN)?;
    Ok(OK)
}

/// `msg` into the caller's `err` (NUL-terminated, cut to `len`), where it gave one
fn say(err: *mut c_char, len: usize, msg: &str) {
    if err.is_null() || len == 0 {
        return;
    }
    let n = msg.len().min(len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(msg.as_ptr(), err as *mut u8, n);
        *err.add(n) = 0;
    }
}

fn push<T>(v: &mut Vec<T>, x: T) -> i64 {
    v.push(x);
    v.len() as i64 - 1
}

/// A context on the system's GPU: its handle, or `ERR_IO` where there is no Metal device.
pub fn open() -> i64 {
    let made = autoreleasepool(|_| {
        let dev = MTLCreateSystemDefaultDevice()?;
        let queue = dev.newCommandQueue()?;
        Some(Ctx {
            dev,
            queue,
            lib: None,
            bufs: Vec::new(),
            psos: Vec::new(),
            icbs: Vec::new(),
        })
    });
    let Some(c) = made else { return ERR_IO as i64 };
    match CTXS.lock() {
        Ok(mut all) => push(&mut all, Some(c)),
        Err(_) => ERR_PANIC as i64,
    }
}

/// The context let go, its buffers, pipelines and command buffers with it; `ERR_DOMAIN` for a second close.
pub fn close(h: i64) -> i32 {
    let Ok(mut all) = CTXS.lock() else {
        return ERR_PANIC;
    };
    match usize::try_from(h).ok().and_then(|i| all.get_mut(i)) {
        Some(slot @ Some(_)) => {
            *slot = None;
            OK
        }
        _ => ERR_DOMAIN,
    }
}

/// A zeroed shared buffer of `len` bytes (16 at least) of the context's own: its index.
pub fn buffer(h: i64, len: u64) -> i64 {
    with(h, |c| {
        let b = c
            .dev
            .newBufferWithLength_options(
                len.max(16) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or(ERR_IO)?;
        unsafe { std::ptr::write_bytes(b.contents().as_ptr() as *mut u8, 0, b.length()) };
        Ok(push(&mut c.bufs, Some(b)))
    })
    .unwrap_or_else(|e| e as i64)
}

/// A buffer over `len` bytes at `ptr` that another allocator owns, not copied: both page-aligned (`ERR_DOMAIN`
/// otherwise), `ERR_IO` where the device declines the range. Its index.
///
/// # Safety
/// `ptr` valid for `len` bytes for as long as the context uses the buffer.
pub unsafe fn alias(h: i64, ptr: *mut u8, len: u64) -> i64 {
    let page = unsafe { libc_page() };
    let Some(p) = NonNull::new(ptr as *mut c_void) else {
        return ERR_NULL as i64;
    };
    if len == 0 || !(ptr as usize).is_multiple_of(page) || !(len as usize).is_multiple_of(page) {
        return ERR_DOMAIN as i64;
    }
    with(h, |c| {
        let b = unsafe {
            c.dev.newBufferWithBytesNoCopy_length_options_deallocator(
                p,
                len as usize,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .ok_or(ERR_IO)?;
        Ok(push(&mut c.bufs, Some(b)))
    })
    .unwrap_or_else(|e| e as i64)
}

/// Buffer `b` let go (an alias: the memory stays its owner's); `ERR_DOMAIN` if it is not held. A command buffer
/// that binds it is freed first ([`free_icb`]): its commands do not keep it.
pub fn free_buffer(h: i64, b: i64) -> i32 {
    with(h, |c| free(&mut c.bufs, b)).unwrap_or_else(|e| e)
}

/// Indirect command buffer `i` let go; `ERR_DOMAIN` if it is not held.
pub fn free_icb(h: i64, i: i64) -> i32 {
    with(h, |c| free(&mut c.icbs, i)).unwrap_or_else(|e| e)
}

unsafe fn libc_page() -> usize {
    extern "C" {
        fn getpagesize() -> i32;
    }
    unsafe { getpagesize() as usize }
}

/// Buffer `b`'s contents and length into `out` and `len`.
///
/// # Safety
/// `out` and `len` writable.
pub unsafe fn contents(h: i64, b: i64, out: *mut *mut u8, len: *mut u64) -> i32 {
    if out.is_null() || len.is_null() {
        return ERR_NULL;
    }
    with(h, |c| {
        let buf = slot(&c.bufs, b)?;
        unsafe {
            *out = buf.contents().as_ptr() as *mut u8;
            *len = buf.length() as u64;
        }
        Ok(OK)
    })
    .unwrap_or_else(|e| e)
}

/// Buffer `b`'s GPU address into `out`: what a kernel dereferences where a table of addresses names the buffer (the
/// command buffers that read it make every buffer of the context resident).
///
/// # Safety
/// `out` writable.
pub unsafe fn gpu_address(h: i64, b: i64, out: *mut u64) -> i32 {
    if out.is_null() {
        return ERR_NULL;
    }
    with(h, |c| {
        let buf = slot(&c.bufs, b)?;
        unsafe { *out = buf.gpuAddress() };
        Ok(OK)
    })
    .unwrap_or_else(|e| e)
}

/// `o`'s math, safe unless `fast`: through `mathMode` where the OS has it (`has_mode`, macOS 15 on), else through
/// `fastMathEnabled`, the one switch before it. Sending `setMathMode:` to macOS 14's options raised an Objective-C
/// exception, which Rust cannot catch: the process aborted at the first compile
fn set_math(o: &MTLCompileOptions, fast: bool, has_mode: bool) {
    if has_mode {
        o.setMathMode(if fast {
            MTLMathMode::Fast
        } else {
            MTLMathMode::Safe
        });
    } else {
        #[allow(deprecated)]
        o.setFastMathEnabled(fast);
    }
}

/// `src` (NUL-terminated UTF-8) compiled as the context's library, the math safe unless `fast` (MLX compiles its
/// kernels safe: the bits a pass gives depend on it). `ERR_PARSE` with the compiler's message in `err`.
///
/// # Safety
/// `src` NUL-terminated; `err` writable for `errlen` bytes (or null).
pub unsafe fn compile(
    h: i64,
    src: *const c_char,
    fast: bool,
    err: *mut c_char,
    errlen: usize,
) -> i32 {
    if src.is_null() {
        return ERR_NULL;
    }
    let Ok(text) = unsafe { CStr::from_ptr(src) }.to_str() else {
        return ERR_UTF8;
    };
    with(h, |c| {
        let o = MTLCompileOptions::new();
        set_math(&o, fast, o.respondsToSelector(sel!(setMathMode:)));
        match c
            .dev
            .newLibraryWithSource_options_error(&NSString::from_str(text), Some(&o))
        {
            Ok(lib) => {
                c.lib = Some(lib);
                Ok(OK)
            }
            Err(e) => {
                say(err, errlen, &e.localizedDescription().to_string());
                Err(ERR_PARSE)
            }
        }
    })
    .unwrap_or_else(|e| e)
}

/// Function `name` of the compiled library as a pipeline an indirect command buffer can run: its index, or
/// `ERR_PARSE` with Metal's message.
///
/// # Safety
/// `name` NUL-terminated; `err` writable for `errlen` bytes (or null).
pub unsafe fn pipeline(h: i64, name: *const c_char, err: *mut c_char, errlen: usize) -> i64 {
    if name.is_null() {
        return ERR_NULL as i64;
    }
    let Ok(n) = unsafe { CStr::from_ptr(name) }.to_str() else {
        return ERR_UTF8 as i64;
    };
    with(h, |c| {
        let lib = c.lib.as_ref().ok_or(ERR_DOMAIN)?;
        let Some(f) = lib.newFunctionWithName(&NSString::from_str(n)) else {
            say(err, errlen, &format!("no function {n}"));
            return Err(ERR_PARSE);
        };
        let d = MTLComputePipelineDescriptor::new();
        d.setComputeFunction(Some(&f));
        d.setSupportIndirectCommandBuffers(true);
        match c
            .dev
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &d,
                MTLPipelineOption::None,
                None,
            ) {
            Ok(p) => Ok(push(&mut c.psos, p)),
            Err(e) => {
                say(err, errlen, &e.localizedDescription().to_string());
                Err(ERR_PARSE)
            }
        }
    })
    .unwrap_or_else(|e| e as i64)
}

/// An indirect command buffer of `cmds` compute commands binding `binds` buffers each: its index.
pub fn icb(h: i64, cmds: u32, binds: u32) -> i64 {
    with(h, |c| {
        let d = MTLIndirectCommandBufferDescriptor::new();
        d.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        d.setInheritBuffers(false);
        d.setInheritPipelineState(false);
        d.setMaxKernelBufferBindCount(binds as usize);
        let b = unsafe {
            c.dev
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &d,
                    cmds as usize,
                    MTLResourceOptions::StorageModeShared,
                )
        }
        .ok_or(ERR_IO)?;
        Ok(push(&mut c.icbs, Some(b)))
    })
    .unwrap_or_else(|e| e as i64)
}

/// Command `k` of indirect command buffer `icb`: pipeline `pso` over gx x gy x gz threadgroups of `threads`, bind j
/// buffer `bufs[j]` at `offs[j]`; `barrier` waits for every command before it. `ERR_DOMAIN` for an index or an
/// offset out of range (nothing encoded).
///
/// # Safety
/// `bufs` and `offs` readable for `nbind` elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cmd(
    h: i64,
    icb: i64,
    k: u32,
    pso: i64,
    nbind: u32,
    bufs: *const i64,
    offs: *const u64,
    grid: [u32; 3],
    threads: u32,
    barrier: bool,
) -> i32 {
    if nbind > 0 && (bufs.is_null() || offs.is_null()) {
        return ERR_NULL;
    }
    let (bufs, offs) = unsafe {
        (
            std::slice::from_raw_parts(bufs, nbind as usize),
            std::slice::from_raw_parts(offs, nbind as usize),
        )
    };
    with(h, |c| {
        let ib = slot(&c.icbs, icb)?;
        let p = idx(&c.psos, pso)?;
        if k as usize >= ib.size() {
            return Err(ERR_DOMAIN);
        }
        let mut bound = Vec::with_capacity(bufs.len());
        for (&b, &o) in bufs.iter().zip(offs) {
            let buf = slot(&c.bufs, b)?;
            if o as usize >= buf.length() {
                return Err(ERR_DOMAIN);
            }
            bound.push((buf, o as usize));
        }
        let cm = unsafe { ib.indirectComputeCommandAtIndex(k as usize) };
        cm.setComputePipelineState(p);
        for (j, (buf, o)) in bound.into_iter().enumerate() {
            unsafe { cm.setKernelBuffer_offset_atIndex(buf, o, j) };
        }
        if barrier {
            cm.setBarrier();
        }
        let size = |x: u32, y: u32, z: u32| MTLSize {
            width: x as usize,
            height: y as usize,
            depth: z as usize,
        };
        cm.concurrentDispatchThreadgroups_threadsPerThreadgroup(
            size(grid[0], grid[1], grid[2]),
            size(threads, 1, 1),
        );
        Ok(OK)
    })
    .unwrap_or_else(|e| e)
}

/// The first `n` commands of `icb` replayed in one command buffer on a concurrent encoder (the commands' barriers
/// order them) and waited for: the GPU's seconds into `gpu_s`, or `ERR_IO` with the command buffer's error in `err`.
///
/// # Safety
/// `gpu_s` writable (or null); `err` writable for `errlen` bytes (or null).
pub unsafe fn run(
    h: i64,
    icb: i64,
    n: u32,
    gpu_s: *mut f64,
    err: *mut c_char,
    errlen: usize,
) -> i32 {
    with(h, |c| {
        let ib = slot(&c.icbs, icb)?;
        if n as usize > ib.size() {
            return Err(ERR_DOMAIN);
        }
        let cb = c.queue.commandBuffer().ok_or(ERR_IO)?;
        let e = cb
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .ok_or(ERR_IO)?;
        for b in c.bufs.iter().flatten() {
            let r: &ProtocolObject<dyn MTLResource> = ProtocolObject::from_ref(&**b);
            e.useResource_usage(r, MTLResourceUsage::Read | MTLResourceUsage::Write);
        }
        unsafe { e.executeCommandsInBuffer_withRange(ib, NSRange::new(0, n as usize)) };
        e.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            let msg = cb.error().map(|e| e.localizedDescription().to_string());
            say(
                err,
                errlen,
                msg.as_deref()
                    .unwrap_or("the command buffer did not complete"),
            );
            return Err(ERR_IO);
        }
        if !gpu_s.is_null() {
            unsafe { *gpu_s = cb.GPUEndTime() - cb.GPUStartTime() };
        }
        Ok(OK)
    })
    .unwrap_or_else(|e| e)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a library compiles through `fastMathEnabled` (macOS 14's switch, kept on every OS since) and, where the OS
    /// answers to it, through `mathMode`: neither raises
    #[test]
    fn compiles_through_either_math_switch() {
        let src = "#include <metal_stdlib>\nkernel void k(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) \
                   { x[i] = metal::exp(x[i]); }";
        let h = open();
        assert!(h >= 0, "no Metal device: {h}");
        let got = with(h, |c| {
            let probe = MTLCompileOptions::new();
            for has_mode in [false, probe.respondsToSelector(sel!(setMathMode:))] {
                let o = MTLCompileOptions::new();
                set_math(&o, false, has_mode);
                c.dev
                    .newLibraryWithSource_options_error(&NSString::from_str(src), Some(&o))
                    .map_err(|_| ERR_PARSE)?;
            }
            Ok(OK)
        });
        assert_eq!(close(h), OK);
        assert_eq!(got, Ok(OK));
    }
}

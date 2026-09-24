// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The page-fenced half of the test scaffolding: a buffer with a guard page on both sides, poisoned
//! slack and optional read-only protection for inputs, on VirtualAlloc/VirtualProtect under Windows
//! and mmap/mprotect elsewhere. The generators, references and packer live in [`refs`] and the
//! unbuffered-read fixture in [`fixture`]; both are re-exported here, so a fenced test sees every
//! helper through `common::*`.
//!
//! `dead_code` is allowed because every test binary compiles this module on its own and uses only
//! the part of it that it needs.
#![allow(dead_code)]

pub mod attn;
pub mod deltak;
pub mod fixture;
pub mod gemvk;
pub mod refs;

// A test binary that never opens a file does not name anything from `fixture`.
#[allow(unused_imports)]
pub use fixture::*;
#[allow(unused_imports)]
pub use refs::*;

pub const SLACK_POISON: u8 = 0xA5;
pub const POISON_I32: i32 = 0x5A5A_5A5A;

/// The access a range of fence pages is set to.
#[derive(Clone, Copy, Debug)]
enum Prot {
    NoAccess,
    ReadOnly,
    ReadWrite,
}

#[cfg(windows)]
mod os {
    use super::Prot;
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(
            addr: *mut c_void,
            size: usize,
            alloc_type: u32,
            protect: u32,
        ) -> *mut c_void;
        fn VirtualProtect(addr: *mut c_void, size: usize, new_protect: u32, old: *mut u32) -> i32;
        fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
        fn GetLastError() -> u32;
    }

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READONLY: u32 = 0x02;
    const PAGE_READWRITE: u32 = 0x04;

    pub fn page() -> usize {
        4096
    }

    /// `total` bytes of committed read-write pages.
    pub fn map(total: usize) -> *mut u8 {
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                total,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as *mut u8;
        assert!(
            !base.is_null(),
            "VirtualAlloc({total}) failed: {}",
            unsafe { GetLastError() }
        );
        base
    }

    /// # Safety
    /// `[p, p + n)` whole pages of one [`map`].
    pub unsafe fn protect(p: *mut u8, n: usize, prot: Prot, label: &str) {
        let flag = match prot {
            Prot::NoAccess => PAGE_NOACCESS,
            Prot::ReadOnly => PAGE_READONLY,
            Prot::ReadWrite => PAGE_READWRITE,
        };
        let mut old = 0u32;
        assert!(
            unsafe { VirtualProtect(p as *mut c_void, n, flag, &mut old) } != 0,
            "{label}: VirtualProtect({prot:?}) failed {}",
            unsafe { GetLastError() }
        );
    }

    /// # Safety
    /// `base` from [`map`], not used again.
    pub unsafe fn unmap(base: *mut u8, _total: usize) {
        unsafe { VirtualFree(base as *mut c_void, 0, MEM_RELEASE) };
    }
}

#[cfg(unix)]
mod os {
    use super::Prot;
    use std::sync::OnceLock;

    /// The OS page (16 KiB on Apple silicon), the granularity mprotect works in.
    pub fn page() -> usize {
        static PAGE: OnceLock<usize> = OnceLock::new();
        *PAGE.get_or_init(|| {
            let n = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            assert!(n > 0, "sysconf(_SC_PAGESIZE) -> {n}");
            n as usize
        })
    }

    /// `total` bytes of private anonymous read-write pages.
    pub fn map(total: usize) -> *mut u8 {
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert!(
            base != libc::MAP_FAILED,
            "mmap({total}) failed: {}",
            std::io::Error::last_os_error()
        );
        base as *mut u8
    }

    /// # Safety
    /// `[p, p + n)` whole pages of one [`map`].
    pub unsafe fn protect(p: *mut u8, n: usize, prot: Prot, label: &str) {
        let flag = match prot {
            Prot::NoAccess => libc::PROT_NONE,
            Prot::ReadOnly => libc::PROT_READ,
            Prot::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
        };
        assert!(
            unsafe { libc::mprotect(p as *mut libc::c_void, n, flag) } == 0,
            "{label}: mprotect({prot:?}) failed {}",
            std::io::Error::last_os_error()
        );
    }

    /// # Safety
    /// `base` and `total` from [`map`], not used again.
    pub unsafe fn unmap(base: *mut u8, total: usize) {
        unsafe { libc::munmap(base as *mut libc::c_void, total) };
    }
}

/// Where the payload sits inside its committed pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    /// The last payload byte is the last byte before the trailing guard page: any write past the
    /// end faults immediately; the slack before the payload is poisoned and checked.
    End,
    /// The first payload byte is the first byte after the leading guard page: any write before
    /// the start faults immediately; the slack after the payload is poisoned and checked.
    Start,
}

/// A payload of `len` elements of `T` between two no-access guard pages.
pub struct Fence<T> {
    base: *mut u8,
    total: usize,
    data: *mut T,
    len: usize,
    align: Align,
    label: String,
}

unsafe impl<T: Send> Send for Fence<T> {}
unsafe impl<T: Sync> Sync for Fence<T> {}

impl<T: Copy> Fence<T> {
    pub fn new(label: &str, len: usize, align: Align) -> Fence<T> {
        let page = os::page();
        let bytes = len * std::mem::size_of::<T>();
        let data_pages = bytes.div_ceil(page).max(1);
        let total = (data_pages + 2) * page;
        let base = os::map(total);
        unsafe {
            os::protect(base, page, Prot::NoAccess, label);
            os::protect(base.add(total - page), page, Prot::NoAccess, label);
            std::ptr::write_bytes(base.add(page), SLACK_POISON, data_pages * page);
        }
        let off = match align {
            Align::End => total - page - bytes,
            Align::Start => page,
        };
        let data = unsafe { base.add(off) } as *mut T;
        Fence {
            base,
            total,
            data,
            len,
            align,
            label: label.to_string(),
        }
    }

    pub fn ptr(&self) -> *const T {
        self.data
    }

    pub fn mut_ptr(&self) -> *mut T {
        self.data
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
    }

    pub fn fill(&mut self, v: T) {
        for x in self.as_mut_slice() {
            *x = v;
        }
    }

    pub fn copy_from(&mut self, src: &[T]) {
        assert_eq!(src.len(), self.len, "{}: copy_from length", self.label);
        self.as_mut_slice().copy_from_slice(src);
    }

    fn data_region(&self) -> (*mut u8, usize) {
        let page = os::page();
        unsafe { (self.base.add(page), self.total - 2 * page) }
    }

    /// Every write into the payload faults from here on.
    pub fn protect_readonly(&self) {
        let (p, n) = self.data_region();
        unsafe { os::protect(p, n, Prot::ReadOnly, &self.label) };
    }

    pub fn protect_readwrite(&self) {
        let (p, n) = self.data_region();
        unsafe { os::protect(p, n, Prot::ReadWrite, &self.label) };
    }

    /// The poisoned slack between the payload and the guard page, as bytes.
    fn slack(&self) -> &[u8] {
        let bytes = self.len * std::mem::size_of::<T>();
        let (p, n) = self.data_region();
        unsafe {
            match self.align {
                Align::End => std::slice::from_raw_parts(p, n - bytes),
                Align::Start => std::slice::from_raw_parts(p.add(bytes), n - bytes),
            }
        }
    }

    /// Panics if any slack byte changed.
    pub fn check_borders(&self, ctx: &str) {
        let s = self.slack();
        if let Some(at) = s.iter().position(|&b| b != SLACK_POISON) {
            let side = match self.align {
                Align::End => format!("{} bytes BEFORE the payload", s.len() - at),
                Align::Start => format!("{} bytes AFTER the payload", at),
            };
            panic!(
                "{ctx}: border write on `{}` ({:?}) at {side}: byte {:#04x} (poison {:#04x})",
                self.label, self.align, s[at], SLACK_POISON
            );
        }
    }
}

impl<T> Drop for Fence<T> {
    fn drop(&mut self) {
        unsafe { os::unmap(self.base, self.total) };
    }
}

/// Forces the ISA the crate selects, before its first kernel call in this process.
pub fn force_isa(name: &str) {
    std::env::set_var("BTB_NATIVE_ISA", name);
}

// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! What the machine's caches hold, read from the OS once, for the kernels' tile sizes. A tile changes which
//! bytes stay in cache while a pass runs, never a sum's order, so none of this moves a single output bit.

use std::sync::OnceLock;

/// The L1 data cache assumed where the OS will not say: the smallest in common use (Zen, Skylake, the
/// E-cores of a hybrid part), so a guess never sizes a tile past what a core can hold.
const L1D_FALLBACK: usize = 32 * 1024;

/// The smallest L1 data cache among the machine's cores, in bytes. The smallest, because the thread pool runs
/// a task on whichever core is free: on a hybrid part the efficiency cores' is the one a tile must fit.
pub fn l1d_bytes() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        l1d_os()
            .filter(|&b| (4 * 1024..=16 * 1024 * 1024).contains(&b))
            .unwrap_or(L1D_FALLBACK)
    })
}

#[cfg(windows)]
fn l1d_os() -> Option<usize> {
    // SYSTEM_LOGICAL_PROCESSOR_INFORMATION and its CACHE_DESCRIPTOR, as winnt.h lays them out
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Cache {
        level: u8,
        associativity: u8,
        line_size: u16,
        size: u32,
        kind: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    union Detail {
        cache: Cache,
        reserved: [u64; 2],
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Info {
        mask: usize,
        relationship: i32,
        detail: Detail,
    }
    const RELATION_CACHE: i32 = 2;
    const CACHE_DATA: i32 = 2;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformation(buffer: *mut Info, length: *mut u32) -> i32;
    }
    let mut len = 0u32;
    // SAFETY: a null buffer asks only for the length the table needs
    unsafe { GetLogicalProcessorInformation(std::ptr::null_mut(), &mut len) };
    let n = len as usize / std::mem::size_of::<Info>();
    if n == 0 {
        return None;
    }
    let empty = Info {
        mask: 0,
        relationship: 0,
        detail: Detail { reserved: [0; 2] },
    };
    let mut buf = vec![empty; n];
    // SAFETY: `buf` holds the `len` bytes the first call asked for
    if unsafe { GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len) } == 0 {
        return None;
    }
    buf.iter()
        .filter(|e| e.relationship == RELATION_CACHE)
        // SAFETY: a RelationCache entry's detail is its cache descriptor
        .map(|e| unsafe { e.detail.cache })
        .filter(|c| c.level == 1 && c.kind == CACHE_DATA)
        .map(|c| c.size as usize)
        .min()
}

#[cfg(target_os = "linux")]
fn l1d_os() -> Option<usize> {
    let size = |v: &str| -> Option<usize> {
        let (n, unit) = match v.as_bytes().last()? {
            b'K' => (&v[..v.len() - 1], 1024),
            b'M' => (&v[..v.len() - 1], 1024 * 1024),
            _ => (v, 1),
        };
        n.parse::<usize>().ok().map(|x| x * unit)
    };
    let mut best: Option<usize> = None;
    for cpu in std::fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = cpu.file_name();
        let name = name.to_string_lossy();
        if name.len() <= 3
            || !name.starts_with("cpu")
            || !name[3..].bytes().all(|c| c.is_ascii_digit())
        {
            continue;
        }
        for index in std::fs::read_dir(cpu.path().join("cache"))
            .into_iter()
            .flatten()
            .flatten()
        {
            let dir = index.path();
            let read = |f: &str| {
                std::fs::read_to_string(dir.join(f))
                    .ok()
                    .map(|v| v.trim().to_string())
            };
            if read("level").as_deref() != Some("1") || read("type").as_deref() != Some("Data") {
                continue;
            }
            if let Some(b) = read("size").as_deref().and_then(size) {
                best = Some(best.map_or(b, |m| m.min(b)));
            }
        }
    }
    best
}

#[cfg(target_os = "macos")]
fn l1d_os() -> Option<usize> {
    use std::ffi::{c_char, c_void, CString};
    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> i32;
    }
    let get = |name: &str| -> Option<usize> {
        let c = CString::new(name).ok()?;
        // an int or a quad: either lands in the low bytes of the zeroed u64
        let mut v = 0u64;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: `v` is writable for `len` bytes and the name is NUL-terminated
        let r = unsafe {
            sysctlbyname(
                c.as_ptr(),
                &mut v as *mut u64 as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (r == 0 && v > 0).then_some(v as usize)
    };
    // each performance level's cores (Apple silicon's P and E), else the machine's single figure
    (0..4)
        .filter_map(|l| get(&format!("hw.perflevel{l}.l1dcachesize")))
        .min()
        .or_else(|| get("hw.l1dcachesize"))
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
fn l1d_os() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_l1d_figure_is_a_plausible_cache() {
        let b = l1d_bytes();
        assert!((4 * 1024..=16 * 1024 * 1024).contains(&b), "{b}");
        assert!(b.is_power_of_two() || b.is_multiple_of(1024), "{b}");
    }
}

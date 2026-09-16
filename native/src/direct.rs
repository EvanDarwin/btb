// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
use crate::codes::{ERR_DOMAIN, ERR_EOF, ERR_IO, ERR_NULL, ERR_PANIC, OK};
use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

pub const SECTOR: u64 = 4096;

pub const DEFAULT_CHUNK: u64 = 16 << 20;

pub const MAX_CHUNK: u64 = 256 << 20;

pub const DEFAULT_DEPTH: usize = 4;

pub const MAX_DEPTH: usize = 32;

#[cfg(windows)]
mod sys {

    use std::ffi::c_void;

    pub type Handle = *mut c_void;

    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 0x0000_0001;
    const FILE_SHARE_WRITE: u32 = 0x0000_0002;
    const OPEN_EXISTING: u32 = 3;

    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
    const ERROR_HANDLE_EOF: u32 = 38;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileW(
            path: *const u16,
            access: u32,
            share: u32,
            sec: *mut c_void,
            disposition: u32,
            flags: u32,
            template: Handle,
        ) -> Handle;
        fn ReadFile(h: Handle, buf: *mut c_void, n: u32, read: *mut u32, ov: *mut c_void) -> i32;
        fn GetFileSizeEx(h: Handle, size: *mut i64) -> i32;
        fn CloseHandle(h: Handle) -> i32;
        fn GetLastError() -> u32;
    }

    // the offset a read carries, so one handle serves any number of threads at once: a handle opened
    // without FILE_FLAG_OVERLAPPED takes the offset from here and still returns only once the read is done
    #[repr(C)]
    struct Overlapped {
        internal: usize,
        internal_high: usize,
        offset: u32,
        offset_high: u32,
        event: Handle,
    }

    fn invalid() -> Handle {
        usize::MAX as Handle
    }

    pub fn last_os_error() -> u32 {
        unsafe { GetLastError() }
    }

    pub struct File(Handle);

    unsafe impl Send for File {}
    unsafe impl Sync for File {}

    impl File {
        pub unsafe fn open(path: *const u16) -> Option<File> {
            unsafe { File::open_shared(path, FILE_SHARE_READ | FILE_SHARE_WRITE) }
        }

        // no other process may write to or delete the file while this handle is held
        pub unsafe fn open_share_read(path: *const u16) -> Option<File> {
            unsafe { File::open_shared(path, FILE_SHARE_READ) }
        }

        unsafe fn open_shared(path: *const u16, share: u32) -> Option<File> {
            let h = unsafe {
                CreateFileW(
                    path,
                    GENERIC_READ,
                    share,
                    std::ptr::null_mut(),
                    OPEN_EXISTING,
                    FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
                    std::ptr::null_mut(),
                )
            };
            if h == invalid() {
                None
            } else {
                Some(File(h))
            }
        }

        pub fn size(&self) -> Option<u64> {
            let mut n: i64 = 0;
            let ok = unsafe { GetFileSizeEx(self.0, &mut n) };
            if ok == 0 || n < 0 {
                None
            } else {
                Some(n as u64)
            }
        }

        pub unsafe fn read_at(&self, buf: *mut u8, n: usize, off: u64) -> Option<usize> {
            let mut ov = Overlapped {
                internal: 0,
                internal_high: 0,
                offset: off as u32,
                offset_high: (off >> 32) as u32,
                event: std::ptr::null_mut(),
            };
            let mut got: u32 = 0;
            let ok = unsafe {
                ReadFile(
                    self.0,
                    buf.cast(),
                    n as u32,
                    &mut got,
                    std::ptr::from_mut(&mut ov).cast(),
                )
            };
            if ok == 0 {
                // a read that starts at the end of the file is the empty read, not a failure: with an
                // OVERLAPPED offset Windows reports it as an error instead of zero bytes
                return (unsafe { GetLastError() } == ERROR_HANDLE_EOF).then_some(0);
            }
            Some(got as usize)
        }
    }

    impl Drop for File {
        fn drop(&mut self) {
            unsafe { CloseHandle(self.0) };
        }
    }
}

#[cfg(unix)]
mod sys {

    use std::fs::File as StdFile;
    use std::os::unix::fs::FileExt;
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    use std::os::unix::fs::OpenOptionsExt;

    // O_DIRECT is per architecture on Linux: 0o40000 on x86-64, 0o200000 on arm64 (where 0o40000 is
    // O_DIRECTORY, which refuses every regular file)
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    const O_DIRECT: i32 = 0o0040000;
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    const O_DIRECT: i32 = 0o0200000;

    pub fn last_os_error() -> u32 {
        std::io::Error::last_os_error().raw_os_error().unwrap_or(0) as u32
    }

    pub struct File(StdFile);

    impl File {
        pub unsafe fn open(path: *const u16) -> Option<File> {
            let s = unsafe { super::utf16_to_string(path) }?;
            let mut o = std::fs::OpenOptions::new();
            o.read(true);
            #[cfg(all(
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ))]
            o.custom_flags(O_DIRECT);
            #[allow(unused_mut)]
            let mut f = o.open(&s).ok()?;
            // a filesystem that does not take O_DIRECT (tmpfs, some FUSE mounts) opens fine and then refuses
            // the first read with EINVAL: probe one sector and fall back to a buffered handle for that file
            #[cfg(all(
                target_os = "linux",
                any(target_arch = "x86_64", target_arch = "aarch64")
            ))]
            {
                if let Some(probe) = super::Scratch::new(super::SECTOR as usize) {
                    let slice = unsafe {
                        std::slice::from_raw_parts_mut(probe.ptr, super::SECTOR as usize)
                    };
                    if let Err(e) = f.read_at(slice, 0) {
                        if e.raw_os_error() == Some(22) {
                            f = std::fs::OpenOptions::new().read(true).open(&s).ok()?;
                        }
                    }
                }
            }
            // macOS has no O_DIRECT; F_NOCACHE keeps the streamed bytes out of the unified buffer cache
            // (the reads are sequential and used once), which on a laptop is the memory the model runs in.
            #[cfg(target_os = "macos")]
            {
                use std::os::unix::io::AsRawFd;
                const F_NOCACHE: i32 = 48;
                unsafe extern "C" {
                    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
                }
                unsafe {
                    fcntl(f.as_raw_fd(), F_NOCACHE, 1i32);
                }
            }
            Some(File(f))
        }

        // Linux's advisory shared lock: a writer that takes the exclusive lock (the checkpoint tools do)
        // is kept off the file while this handle is held. Nothing else changes if the lock is refused.
        pub unsafe fn open_share_read(path: *const u16) -> Option<File> {
            let f = unsafe { File::open(path) }?;
            #[cfg(target_os = "linux")]
            {
                use std::os::unix::io::AsRawFd;
                const LOCK_SH: i32 = 1;
                const LOCK_NB: i32 = 4;
                unsafe extern "C" {
                    fn flock(fd: i32, op: i32) -> i32;
                }
                unsafe {
                    let _ = flock(f.0.as_raw_fd(), LOCK_SH | LOCK_NB);
                }
            }
            Some(f)
        }

        pub fn size(&self) -> Option<u64> {
            self.0.metadata().ok().map(|m| m.len())
        }

        pub unsafe fn read_at(&self, buf: *mut u8, n: usize, off: u64) -> Option<usize> {
            let slice = unsafe { std::slice::from_raw_parts_mut(buf, n) };
            // a signal landing during a slow read interrupts `pread` with no bytes moved; Python leaves
            // signals interruptible, so that is a retry, not a failed pass
            loop {
                match self.0.read_at(slice, off) {
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    r => return r.ok(),
                }
            }
        }
    }
}

#[cfg(not(any(windows, unix)))]
compile_error!("the reader needs Windows or unix");

pub fn last_os_error() -> u32 {
    sys::last_os_error()
}

struct Scratch {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: the buffer is owned by whoever holds the struct and freed once, in `drop`
unsafe impl Send for Scratch {}

impl Scratch {
    fn new(len: usize) -> Option<Scratch> {
        let layout = Layout::from_size_align(len, SECTOR as usize).ok()?;
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            None
        } else {
            Some(Scratch { ptr, layout })
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) };
    }
}

thread_local! {
    static SCRATCH: RefCell<Option<Scratch>> = const { RefCell::new(None) };
}

/// Runs `f` on this thread's bounce buffer, at least `len` bytes and sector-aligned, kept for the
/// thread's next call; `None` if the allocation failed.
fn with_scratch<R>(len: usize, f: impl FnOnce(*mut u8) -> R) -> Option<R> {
    SCRATCH.with(|cell| {
        let mut held = cell.borrow_mut();
        if held.as_ref().is_none_or(|s| s.layout.size() < len) {
            *held = None;
            *held = Scratch::new(len);
        }
        held.as_ref().map(|s| f(s.ptr))
    })
}

#[derive(Clone, Copy)]
struct Dst(*mut u8);

// SAFETY: the workers write disjoint byte ranges of the destination (each chunk its own), and the caller
// does not write it during the call
unsafe impl Send for Dst {}
unsafe impl Sync for Dst {}

impl Dst {
    /// # Safety
    /// `at` inside the destination.
    #[inline]
    unsafe fn at(self, at: u64) -> *mut u8 {
        unsafe { self.0.add(at as usize) }
    }
}

#[cfg(not(windows))]
unsafe fn utf16_to_string(path: *const u16) -> Option<String> {
    let mut n = 0usize;
    while unsafe { *path.add(n) } != 0 {
        n += 1;
        if n > 1 << 20 {
            return None;
        }
    }
    String::from_utf16(unsafe { std::slice::from_raw_parts(path, n) }).ok()
}

pub fn wide(path: &std::path::Path) -> Vec<u16> {
    #[cfg(windows)]
    let mut v: Vec<u16> = {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().collect()
    };
    #[cfg(not(windows))]
    let mut v: Vec<u16> = path.to_string_lossy().encode_utf16().collect();
    v.push(0);
    v
}

#[inline]
fn round_up(v: u64, to: u64) -> Option<u64> {
    v.checked_add(to - 1).map(|x| x & !(to - 1))
}

/// The sector span `[a0, a1)` covering `[off, off + len)`, split into `nchunks` chunks over `workers`
/// threads. `Err` is the `codes` value the call fails with.
struct Plan {
    a0: u64,
    a1: u64,
    end: u64,
    chunk: u64,
    nchunks: usize,
    workers: usize,
}

fn plan(off: u64, len: u64, chunk_bytes: u64, depth: usize) -> Result<Plan, i32> {
    let end = off.checked_add(len).ok_or(ERR_DOMAIN)?;
    let a0 = off & !(SECTOR - 1);
    let a1 = round_up(end, SECTOR).ok_or(ERR_DOMAIN)?;
    let chunk = if chunk_bytes == 0 {
        DEFAULT_CHUNK
    } else {
        round_up(chunk_bytes.min(MAX_CHUNK), SECTOR)
            .ok_or(ERR_DOMAIN)?
            .clamp(SECTOR, MAX_CHUNK)
    };
    let nchunks = (a1 - a0).div_ceil(chunk) as usize;
    let depth = if depth == 0 { DEFAULT_DEPTH } else { depth };
    Ok(Plan {
        a0,
        a1,
        end,
        chunk,
        nchunks,
        workers: depth.clamp(1, MAX_DEPTH).min(nchunks),
    })
}

/// Reads `want` bytes at `at` into `buf`, retrying a short read; stops at the file's last partial
/// sector (an unbuffered read there returns fewer bytes than the sector asked for). `None` is the
/// OS refusing.
///
/// # Safety
/// `buf` writable for `want` bytes.
unsafe fn fill(file: &sys::File, buf: *mut u8, want: u64, at: u64) -> Option<u64> {
    let mut got: u64 = 0;
    while got < want {
        let n =
            unsafe { file.read_at(buf.add(got as usize), (want - got) as usize, at + got) }? as u64;
        if n == 0 {
            break;
        }
        got += n;

        if !got.is_multiple_of(SECTOR) {
            break;
        }
    }
    Some(got)
}

#[inline]
fn fail(err: &AtomicI32, code: i32) {
    let _ = err.compare_exchange(OK, code, Ordering::AcqRel, Ordering::Relaxed);
}

/// # Safety
/// `path` NUL-terminated UTF-16; `dst` writable for `len` bytes and not written concurrently.
pub unsafe fn read_direct(
    path: *const u16,
    off: u64,
    len: u64,
    dst: *mut u8,
    chunk_bytes: u64,
    depth: usize,
) -> i32 {
    if len == 0 {
        return OK;
    }
    if dst.is_null() || path.is_null() {
        return ERR_NULL;
    }
    let p = match plan(off, len, chunk_bytes, depth) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let file = match unsafe { sys::File::open(path) } {
        Some(f) => f,
        None => return ERR_IO,
    };
    match file.size() {
        Some(sz) if p.end <= sz => {}
        Some(_) => return ERR_EOF,
        None => return ERR_IO,
    }
    unsafe { read_span(&file, &p, off, Dst(dst)) }
}

/// The planned span of `file` into `dst`: `p.workers` threads take chunks in turn, each read straight into
/// the destination when the span and the destination are sector-aligned, else into the thread's bounce
/// buffer and copied. The first error stops every worker; `OK` or its code is returned.
///
/// # Safety
/// `dst` writable for the span's bytes and not written concurrently; `p` planned for `off` and `dst`'s length.
unsafe fn read_span(file: &sys::File, p: &Plan, off: u64, dst: Dst) -> i32 {
    let len = p.end - off;
    let straight = off.is_multiple_of(SECTOR)
        && len.is_multiple_of(SECTOR)
        && (dst.0 as usize).is_multiple_of(SECTOR as usize);
    let bounce = p.chunk.min(p.a1 - p.a0) as usize;
    let next = AtomicUsize::new(0);
    let err = AtomicI32::new(OK);

    let worker = || loop {
        if err.load(Ordering::Relaxed) != OK {
            return;
        }
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= p.nchunks {
            return;
        }
        let c0 = p.a0 + (i as u64) * p.chunk;
        let want = p.chunk.min(p.a1 - c0);

        let s = c0.max(off);
        let e = (c0 + want).min(p.end);
        let needed = if e > s { e - c0 } else { 0 };

        let got = if straight {
            unsafe { fill(file, dst.at(c0 - off), want, c0) }
        } else {
            with_scratch(bounce, |buf| {
                let got = unsafe { fill(file, buf, want, c0) };
                if got.is_some_and(|g| g >= needed) && e > s {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            buf.add((s - c0) as usize),
                            dst.at(s - off),
                            (e - s) as usize,
                        )
                    };
                }
                got
            })
            .flatten()
        };
        match got {
            Some(g) if g >= needed => {}
            Some(_) => {
                fail(&err, ERR_EOF);
                return;
            }
            None => {
                fail(&err, ERR_IO);
                return;
            }
        }
    };

    let guarded = || {
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(&worker)).is_err() {
            fail(&err, ERR_PANIC);
        }
    };

    // the extra workers come from the cached pool for this width (their bounce buffers persist with them); a
    // pool that cannot be built falls back to threads of the call
    match (p.workers > 1)
        .then(|| crate::gemv::pool_for(p.workers - 1))
        .flatten()
    {
        Some(pool) => pool.scope(|scope| {
            for _ in 1..p.workers {
                scope.spawn(|_| guarded());
            }
            guarded();
        }),
        None => std::thread::scope(|scope| {
            for _ in 1..p.workers {
                scope.spawn(guarded);
            }
            guarded();
        }),
    }

    err.load(Ordering::Acquire)
}

struct Open {
    file: sys::File,
    size: u64,
}

static NEXT_HANDLE: AtomicI64 = AtomicI64::new(1);
static OPEN: LazyLock<Mutex<HashMap<i64, Arc<Open>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn table() -> MutexGuard<'static, HashMap<i64, Arc<Open>>> {
    OPEN.lock().unwrap_or_else(|e| e.into_inner())
}

/// # Safety
/// `path` NUL-terminated UTF-16.
pub unsafe fn open(path: *const u16) -> i64 {
    if path.is_null() {
        return ERR_NULL as i64;
    }
    let file = match unsafe { sys::File::open_share_read(path) } {
        Some(f) => f,
        None => return ERR_IO as i64,
    };
    let size = match file.size() {
        Some(s) => s,
        None => return ERR_IO as i64,
    };
    let id = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    table().insert(id, Arc::new(Open { file, size }));
    id
}

pub fn close(handle: i64) -> i32 {
    // the entry leaves the table under the lock; the file itself closes once the last read holding it returns
    let gone = table().remove(&handle);
    match gone {
        Some(_) => OK,
        None => ERR_DOMAIN,
    }
}

/// # Safety
/// `dst` writable for `len` bytes and not written concurrently.
pub unsafe fn read_at(
    handle: i64,
    off: u64,
    len: u64,
    dst: *mut u8,
    chunk_bytes: u64,
    depth: usize,
) -> i32 {
    if len == 0 {
        return OK;
    }
    if dst.is_null() {
        return ERR_NULL;
    }
    let p = match plan(off, len, chunk_bytes, depth) {
        Ok(p) => p,
        Err(code) => return code,
    };
    let open = match table().get(&handle).cloned() {
        Some(o) => o,
        None => return ERR_DOMAIN,
    };
    if p.end > open.size {
        return ERR_EOF;
    }
    unsafe { read_span(&open.file, &p, off, Dst(dst)) }
}

#[cfg(all(test, windows))]
mod tests {

    use super::*;

    const ERROR_INVALID_PARAMETER: u32 = 87;

    #[test]
    fn unbuffered_flag_is_actually_set() {
        let dir = std::env::var_os("BTB_DIRECT_TEST_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = dir.join(format!("btb_direct_flagprobe_{}.bin", std::process::id()));
        std::fs::write(&path, vec![7u8; 64 * 1024]).expect("write probe file");
        let w = wide(&path);

        let f = unsafe { sys::File::open(w.as_ptr()) }.expect("open");
        let scratch = Scratch::new(64 * 1024).expect("scratch");

        let ok = unsafe { f.read_at(scratch.ptr, SECTOR as usize, 0) };
        assert_eq!(ok, Some(SECTOR as usize), "aligned read failed");

        let bad = unsafe { f.read_at(scratch.ptr, 4095, 1) };
        let os = last_os_error();
        drop(f);
        let _ = std::fs::remove_file(&path);

        assert!(
            bad.is_none() && os == ERROR_INVALID_PARAMETER,
            "a misaligned unbuffered read was accepted (got {bad:?}, os error {os}): \
             FILE_FLAG_NO_BUFFERING is not in effect and every read is going \
             through the page cache"
        );
    }
}

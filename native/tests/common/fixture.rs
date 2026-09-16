// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! A temporary file of known bytes for the unbuffered-read tests, and the pattern it holds. It
//! touches only `std::fs` and the crate's own entry points, so the cross-platform test files
//! include it with `#[path = "common/fixture.rs"] mod fixture;` while the fenced tests reach it
//! through `common`.
//!
//! `dead_code` is allowed because every test binary compiles this module on its own: a binary that
//! never opens a handle does not call `Fixture::open`.
#![allow(dead_code)]

use btb_native::btb_open;
use btb_native::direct::{self, wide};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// splitmix64 over the block index: eight bytes per block, so any byte of the file is predictable
/// from its offset and a wrong sector is obvious in the first difference.
pub fn pattern(len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    for (i, blk) in v.chunks_mut(8).enumerate() {
        let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF_CAFE_F00D;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        blk.copy_from_slice(&z.to_le_bytes()[..blk.len()]);
    }
    v
}

static SERIAL: AtomicU64 = AtomicU64::new(0);

/// A file of `len` bytes of [`pattern`] under `BTB_DIRECT_TEST_DIR` (the system temp directory by
/// default), removed when the fixture drops. `bytes` is what the file holds, so a read can be
/// compared without reading it back.
pub struct Fixture {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub wide: Vec<u16>,
}

impl Fixture {
    pub fn new(name: &str, len: u64) -> Fixture {
        let dir = std::env::var_os("BTB_DIRECT_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&dir).expect("fixture dir");
        let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!(
            "btb_fixture_{name}_{}_{serial}.bin",
            std::process::id()
        ));
        let bytes = pattern(len as usize);
        std::fs::write(&path, &bytes).expect("write fixture");
        let wide = wide(&path);
        Fixture { path, bytes, wide }
    }

    /// The file open for unbuffered reads; panics if the open is refused.
    pub fn open(&self) -> i64 {
        let h = unsafe { btb_open(self.wide.as_ptr()) };
        assert!(
            h >= 0,
            "btb_open -> {h} (os error {})",
            direct::last_os_error()
        );
        h
    }

    /// The bytes the file holds over `[off, off + len)`.
    pub fn span(&self, off: u64, len: u64) -> &[u8] {
        &self.bytes[off as usize..(off + len) as usize]
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

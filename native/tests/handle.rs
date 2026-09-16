// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The open handle: `btb_read_at` against `btb_read_direct` byte for byte over both of its paths (the
//! sector-aligned read into the destination and the bounce buffer), sixteen threads on one handle, and
//! the codes for a missing file, a handle that is not open and an empty read.

use btb_native::codes::*;
use btb_native::direct::{self, wide, SECTOR};
use btb_native::{btb_close, btb_open, btb_read_at, btb_read_direct};
use std::path::Path;

#[path = "common/fixture.rs"]
mod fixture;

use fixture::Fixture;

const MIB: u64 = 1 << 20;

/// A destination whose first payload byte sits `skew` bytes past a sector boundary: `skew == 0` is the
/// alignment that lets the read land in it with no bounce buffer. This is the cross-platform stand-in
/// for the guard-page fence: the fill outside the payload plays the part of the poisoned slack.
struct Dest {
    buf: Vec<u8>,
    at: usize,
    len: usize,
}

impl Dest {
    fn new(len: u64, skew: usize) -> Dest {
        let sector = SECTOR as usize;
        let buf = vec![0xA5u8; len as usize + 2 * sector];
        let pad = (sector - (buf.as_ptr() as usize % sector)) % sector;
        Dest {
            buf,
            at: pad + skew,
            len: len as usize,
        }
    }

    fn ptr(&mut self) -> *mut u8 {
        unsafe { self.buf.as_mut_ptr().add(self.at) }
    }

    fn payload(&self) -> &[u8] {
        &self.buf[self.at..self.at + self.len]
    }

    /// every byte outside the payload is still the fill
    fn check_slack(&self, ctx: &str) {
        for (i, &b) in self.buf.iter().enumerate() {
            if (i < self.at || i >= self.at + self.len) && b != 0xA5 {
                panic!("{ctx}: byte {i} outside the destination was written ({b:#04x})");
            }
        }
    }
}

fn check(f: &Fixture, h: i64, off: u64, len: u64, chunk: u64, depth: usize, skew: usize) {
    let ctx = format!("read_at(off={off}, len={len}, chunk={chunk}, depth={depth}, skew={skew})");
    let mut dst = Dest::new(len, skew);
    let rc = unsafe { btb_read_at(h, off, len, dst.ptr(), chunk, depth) };
    assert_eq!(
        rc,
        OK,
        "{ctx} -> {rc} (os error {})",
        direct::last_os_error()
    );
    dst.check_slack(&ctx);

    let want = f.span(off, len);
    if dst.payload() != want {
        let at = dst
            .payload()
            .iter()
            .zip(want)
            .position(|(a, b)| a != b)
            .unwrap_or(want.len());
        panic!(
            "{ctx}: first difference at byte {at} (file offset {}, sector {})",
            off + at as u64,
            (off + at as u64) / SECTOR
        );
    }

    let mut direct_buf = vec![0u8; len as usize];
    let rc = unsafe { btb_read_direct(f.wide.as_ptr(), off, len, direct_buf.as_mut_ptr(), chunk) };
    assert_eq!(rc, OK, "{ctx}: btb_read_direct -> {rc}");
    assert_eq!(
        dst.payload(),
        &direct_buf[..],
        "{ctx}: differs from btb_read_direct"
    );
}

/// Ranges at offsets and destination alignments on and off a sector boundary, and the file's
/// partial last sector, all match what `btb_read_direct` returns for the same range.
#[test]
fn ranges_match_read_direct() {
    const FILE_LEN: u64 = 40 * MIB + 777;
    let f = Fixture::new("handle_ranges", FILE_LEN);
    let h = f.open();

    for &off in &[0u64, 1, 4095, 4096, 8192, MIB + 7] {
        for &len in &[1u64, 4095, 4096, 8192, MIB + 3, 20 * MIB] {
            for skew in [0usize, 1, 512] {
                check(&f, h, off, len, 0, 0, skew);
            }
        }
    }
    // the tail, where the file's last sector is partial
    for &len in &[1u64, 777, 4096, MIB] {
        check(&f, h, FILE_LEN - len, len, 0, 0, 0);
        check(&f, h, FILE_LEN - len, len, 4096, 3, 1);
    }
    assert_eq!(btb_close(h), OK);
}

/// The chunk size and the worker depth are scheduling choices: every combination delivers the same
/// bytes as `btb_read_direct`.
#[test]
fn chunks_and_depth_do_not_change_the_bytes() {
    const FILE_LEN: u64 = 24 * MIB + 111;
    let f = Fixture::new("handle_chunks", FILE_LEN);
    let h = f.open();

    for &chunk in &[1u64, 4096, 5000, 12288, 64 * 1024, MIB] {
        for &depth in &[1usize, 4, 16] {
            for &(off, len) in &[
                (0u64, 4097u64),
                (1, 4095),
                (4095, 4098),
                (4096, 4096),
                (8192, 4 * MIB),
                (MIB + 7, 3 * MIB + 13),
                (FILE_LEN - 12288 - 5, 12288 + 5),
            ] {
                check(&f, h, off, len, chunk, depth, 0);
                check(&f, h, off, len, chunk, depth, 7);
            }
        }
    }
    assert_eq!(btb_close(h), OK);
}

/// Sixteen threads reading their own ranges through one handle at once each get their own bytes,
/// and none of them writes outside its destination.
#[test]
fn sixteen_threads_read_one_handle() {
    const FILE_LEN: u64 = 64 * MIB;
    const THREADS: u64 = 16;
    let f = Fixture::new("handle_threads", FILE_LEN);
    let h = f.open();

    // each thread walks its own ranges, aligned and not, over the whole file for 200 reads apiece
    std::thread::scope(|scope| {
        for t in 0..THREADS {
            let f = &f;
            scope.spawn(move || {
                let mut seed = t.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
                for i in 0..200u64 {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let len = 1 + (seed % (2 * MIB));
                    let off = (seed >> 20) % (FILE_LEN - len);
                    let (off, len) = if i % 2 == 0 {
                        (off & !(SECTOR - 1), len.next_multiple_of(SECTOR))
                    } else {
                        (off, len)
                    };
                    let len = len.min(FILE_LEN - off);
                    let mut dst = Dest::new(len, (i % 3) as usize);
                    let rc = unsafe { btb_read_at(h, off, len, dst.ptr(), 0, 1) };
                    assert_eq!(rc, OK, "thread {t} read {i}: off={off} len={len} -> {rc}");
                    assert_eq!(
                        dst.payload(),
                        f.span(off, len),
                        "thread {t} read {i}: off={off} len={len} came back wrong"
                    );
                    dst.check_slack("threaded");
                }
            });
        }
    });
    assert_eq!(btb_close(h), OK);
}

/// A missing path, a range past the end, an overflowing range, a null destination, a handle that
/// was never open and one that has been closed each return their own code.
#[test]
fn error_codes() {
    const FILE_LEN: u64 = 4 * MIB + 5;
    let f = Fixture::new("handle_errors", FILE_LEN);
    let h = f.open();
    let mut buf = vec![0u8; (FILE_LEN + 8192) as usize];

    let missing = wide(Path::new("Z:\\no\\such\\dir\\file.bin"));
    assert_eq!(unsafe { btb_open(missing.as_ptr()) }, ERR_IO as i64);
    assert_eq!(unsafe { btb_open(std::ptr::null()) }, ERR_NULL as i64);

    for &(off, len) in &[
        (FILE_LEN - 10, 100u64),
        (FILE_LEN, 1),
        (FILE_LEN + 4096, 4096),
        (0, FILE_LEN + 1),
    ] {
        assert_eq!(
            unsafe { btb_read_at(h, off, len, buf.as_mut_ptr(), 0, 0) },
            ERR_EOF,
            "off={off} len={len} should be past the end"
        );
    }

    assert_eq!(
        unsafe { btb_read_at(h, u64::MAX - 3, 8, buf.as_mut_ptr(), 0, 0) },
        ERR_DOMAIN
    );
    assert_eq!(
        unsafe { btb_read_at(h, 0, 4096, std::ptr::null_mut(), 0, 0) },
        ERR_NULL
    );

    // len == 0 is OK on any handle, open or not, and writes nothing
    assert_eq!(
        unsafe { btb_read_at(h, 12345, 0, std::ptr::null_mut(), 0, 0) },
        OK
    );
    assert_eq!(unsafe { btb_read_at(-1, 0, 0, buf.as_mut_ptr(), 0, 0) }, OK);

    // a handle that was never open, and one that has been closed
    assert_eq!(
        unsafe { btb_read_at(0, 0, 4096, buf.as_mut_ptr(), 0, 0) },
        ERR_DOMAIN
    );
    assert_eq!(btb_close(0), ERR_DOMAIN);
    assert_eq!(btb_close(-7), ERR_DOMAIN);
    assert_eq!(btb_close(h), OK);
    assert_eq!(btb_close(h), ERR_DOMAIN, "a second close");
    assert_eq!(
        unsafe { btb_read_at(h, 0, 4096, buf.as_mut_ptr(), 0, 0) },
        ERR_DOMAIN,
        "a closed handle"
    );
}

/// Two handles on two files read their own bytes, and closing one leaves the other open.
#[test]
fn handles_are_independent() {
    let a = Fixture::new("handle_indep_a", 2 * MIB);
    let b = Fixture::new("handle_indep_b", 3 * MIB + 9);
    let (ha, hb) = (a.open(), b.open());
    assert_ne!(ha, hb);

    for (f, h) in [(&a, ha), (&b, hb)] {
        let mut dst = Dest::new(MIB, 5);
        assert_eq!(unsafe { btb_read_at(h, 4095, MIB, dst.ptr(), 0, 0) }, OK);
        assert_eq!(dst.payload(), f.span(4095, MIB));
    }
    assert_eq!(btb_close(ha), OK);
    // closing one leaves the other open
    let mut dst = Dest::new(4096, 0);
    assert_eq!(unsafe { btb_read_at(hb, 0, 4096, dst.ptr(), 0, 0) }, OK);
    assert_eq!(dst.payload(), b.span(0, 4096));
    assert_eq!(btb_close(hb), OK);
}

// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Unbuffered reads by path: every byte `btb_read_direct` delivers is compared to the bytes the
//! fixture wrote, over awkward offsets, lengths, chunk sizes, worker depths and destination
//! alignments, plus the codes a refused read returns.

use btb_native::btb_read_direct;
use btb_native::codes::*;
use btb_native::direct::{self, wide, MAX_CHUNK, SECTOR};
use std::path::Path;

#[path = "common/fixture.rs"]
mod fixture;

use fixture::Fixture;

const MIB: u64 = 1 << 20;

/// One read into a buffer whose payload starts `skew` bytes in, compared byte for byte.
fn check(f: &Fixture, off: u64, len: u64, chunk: u64, skew: usize) {
    let mut buf = vec![0xA5u8; len as usize + skew];
    let rc =
        unsafe { btb_read_direct(f.wide.as_ptr(), off, len, buf.as_mut_ptr().add(skew), chunk) };
    assert_eq!(
        rc,
        OK,
        "btb_read_direct(off={off}, len={len}, chunk={chunk}) -> {rc} (os error {})",
        direct::last_os_error()
    );
    let want = f.span(off, len);
    let got = &buf[skew..];
    if got != want {
        let at = got
            .iter()
            .zip(want)
            .position(|(a, b)| a != b)
            .unwrap_or(want.len());
        panic!(
            "off={off} len={len} chunk={chunk} skew={skew}: first difference at byte {at} \
             (file offset {}, sector {}, chunk {}): got {:?} want {:?}",
            off + at as u64,
            (off + at as u64) / SECTOR,
            at as u64 / chunk.max(1),
            &got[at..(at + 8).min(got.len())],
            &want[at..(at + 8).min(want.len())],
        );
    }
}

/// Ranges from one byte to a hundred megabytes, at offsets on and off a sector boundary and into
/// destinations on and off one, come back as the bytes the fixture wrote.
#[test]
fn ranges_match_the_bytes_written() {
    const FILE_LEN: u64 = 106 * MIB + 777;
    let f = Fixture::new("direct_ranges", FILE_LEN);

    let lens = [1u64, 4095, 4096, 16 * MIB + 3, 100 * MIB];

    let offs = [0u64, 1, 4095, 4096, MIB + 7];

    for &off in &offs {
        for &len in &lens {
            assert!(off + len <= FILE_LEN);
            check(&f, off, len, 0, 0);
            check(&f, off, len, 0, 3);
        }
    }

    for &len in &[1u64, 777, 4096, MIB] {
        check(&f, FILE_LEN - len, len, 0, 0);
        check(&f, FILE_LEN - len, len, 4096, 1);
    }
}

/// Cutting the same span into chunks, including chunks smaller than a sector and chunks that do
/// not divide it, does not change a byte.
#[test]
fn chunk_boundaries() {
    const FILE_LEN: u64 = 20 * MIB + 111;
    let f = Fixture::new("direct_chunks", FILE_LEN);

    for &chunk in &[1u64, 4096, 5000, 8192, 12288, 64 * 1024, MIB] {
        for &(off, len) in &[
            (0u64, 4097u64),
            (1, 4095),
            (4095, 4098),
            (4096, 4096),
            (MIB + 7, 3 * MIB + 13),
            (12288 - 1, 12288 + 1),
        ] {
            check(&f, off, len, chunk, 0);
            check(&f, off, len, chunk, 3);
        }
    }

    check(&f, 4095, 16 * MIB + 3, 4096, 3);
}

/// A chunk size above the cap is clamped to it, not refused: the read succeeds and delivers the
/// same bytes as the default chunk size.
#[test]
fn an_oversized_chunk_is_clamped_not_refused() {
    const FILE_LEN: u64 = 8 * MIB + 33;
    let f = Fixture::new("direct_bigchunk", FILE_LEN);

    for &chunk in &[MAX_CHUNK, MAX_CHUNK + 1, 1 << 40, u64::MAX] {
        check(&f, 4095, 6 * MIB + 7, chunk, 0);
        check(&f, 0, FILE_LEN, chunk, 3);
    }
}

/// The worker count is a scheduling choice: one thread and sixty-four deliver the same bytes.
#[test]
fn depth_does_not_change_bytes() {
    const FILE_LEN: u64 = 24 * MIB + 33;
    let f = Fixture::new("direct_depth", FILE_LEN);

    for &depth in &[1usize, 2, 3, 4, 8, 64] {
        for &(off, len, chunk) in &[
            (0u64, 20 * MIB, 0u64),
            (MIB + 7, 8 * MIB + 5, MIB),
            (4095, 4097, 4096),
            (FILE_LEN - 33, 33, 0),
        ] {
            let mut buf = vec![0u8; len as usize + 3];
            let rc = unsafe {
                direct::read_direct(
                    f.wide.as_ptr(),
                    off,
                    len,
                    buf.as_mut_ptr().add(3),
                    chunk,
                    depth,
                )
            };
            assert_eq!(rc, OK, "depth={depth} off={off} len={len}");
            assert_eq!(
                &buf[3..],
                f.span(off, len),
                "depth={depth} off={off} len={len} chunk={chunk} changed the bytes"
            );
        }
    }
}

/// A missing file, a range past the end, an overflowing range and a null pointer each return their
/// own code; a zero-length read is `OK` and does not open the file.
#[test]
fn error_codes() {
    const FILE_LEN: u64 = 4 * MIB + 5;
    let f = Fixture::new("direct_errors", FILE_LEN);

    let mut buf = vec![0u8; (FILE_LEN + 8192) as usize];

    let missing = wide(Path::new("Z:\\no\\such\\dir\\file.bin"));
    assert_eq!(
        unsafe { btb_read_direct(missing.as_ptr(), 0, 4096, buf.as_mut_ptr(), 0) },
        ERR_IO
    );

    for &(off, len) in &[
        (FILE_LEN - 10, 100u64),
        (FILE_LEN, 1),
        (FILE_LEN + 4096, 4096),
        (0, FILE_LEN + 1),
    ] {
        assert_eq!(
            unsafe { btb_read_direct(f.wide.as_ptr(), off, len, buf.as_mut_ptr(), 0) },
            ERR_EOF,
            "off={off} len={len} should be past the end"
        );
    }

    assert_eq!(
        unsafe { btb_read_direct(f.wide.as_ptr(), u64::MAX - 3, 8, buf.as_mut_ptr(), 0) },
        ERR_DOMAIN
    );

    assert_eq!(
        unsafe { btb_read_direct(std::ptr::null(), 0, 4096, buf.as_mut_ptr(), 0) },
        ERR_NULL
    );
    assert_eq!(
        unsafe { btb_read_direct(f.wide.as_ptr(), 0, 4096, std::ptr::null_mut(), 0) },
        ERR_NULL
    );

    assert_eq!(
        unsafe { btb_read_direct(f.wide.as_ptr(), 0, 0, std::ptr::null_mut(), 0) },
        OK
    );
    assert_eq!(
        unsafe { btb_read_direct(missing.as_ptr(), 12345, 0, buf.as_mut_ptr(), 0) },
        OK
    );
}

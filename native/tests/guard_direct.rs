// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced unbuffered reads: the destination sits against a guard page (end and start), so a
//! write past `dst + len` or before `dst` faults; every byte is compared to the file. `btb_read_at`
//! runs the same cases on an open handle, over the aligned path (which hands the destination itself to
//! the drive) and the bounce path.
#![cfg(windows)]

mod common;

use btb_native::codes::OK;
use btb_native::direct;
use btb_native::{btb_close, btb_open, btb_read_at, btb_read_direct};
use common::*;

const MIB: u64 = 1 << 20;

fn verify(f: &Fixture, dst: &Fence<u8>, off: u64, len: u64, ctx: &str) {
    dst.check_borders(ctx);
    let want = f.span(off, len);
    if dst.as_slice() != want {
        let at = dst
            .as_slice()
            .iter()
            .zip(want)
            .position(|(a, b)| a != b)
            .unwrap_or(want.len());
        panic!(
            "{ctx}: first difference at byte {at} (file offset {})",
            off + at as u64
        );
    }
}

fn check(f: &Fixture, off: u64, len: u64, chunk: u64, align: Align) {
    let mut dst = Fence::<u8>::new("dst", len as usize, align);
    dst.fill(0x3C);
    let rc = unsafe { btb_read_direct(f.wide.as_ptr(), off, len, dst.mut_ptr(), chunk) };
    let ctx = format!("read off={off} len={len} chunk={chunk} {align:?}");
    assert_eq!(
        rc,
        OK,
        "{ctx}: rc {rc} (os error {})",
        direct::last_os_error()
    );
    verify(f, &dst, off, len, &ctx);
}

fn check_at(f: &Fixture, h: i64, off: u64, len: u64, chunk: u64, depth: usize, align: Align) {
    let mut dst = Fence::<u8>::new("dst", len as usize, align);
    dst.fill(0x3C);
    let rc = unsafe { btb_read_at(h, off, len, dst.mut_ptr(), chunk, depth) };
    let path = if off.is_multiple_of(4096)
        && len.is_multiple_of(4096)
        && (dst.mut_ptr() as usize).is_multiple_of(4096)
    {
        "aligned"
    } else {
        "bounce"
    };
    let ctx = format!("read_at off={off} len={len} chunk={chunk} depth={depth} {align:?} ({path})");
    assert_eq!(
        rc,
        OK,
        "{ctx}: rc {rc} (os error {})",
        direct::last_os_error()
    );
    verify(f, &dst, off, len, &ctx);
}

/// Every awkward offset and length, at seven chunk sizes and both alignments: the bytes match the
/// file and nothing outside the destination is touched.
#[test]
fn every_edge_case_against_the_guard_pages() {
    const FILE_LEN: u64 = 6 * MIB + 777;
    let f = Fixture::new("guard_edges", FILE_LEN);
    let cases: &[(u64, u64)] = &[
        (0, 1),
        (1, 1),
        (0, 4095),
        (1, 4095),
        (4095, 1),
        (4095, 2),
        (4095, 4098),
        (4096, 4096),
        (4096, 4097),
        (12287, 12289),
        (MIB + 7, 3 * MIB + 13),
        (0, 2 * MIB),
        (777, 2 * MIB + 1),
        (FILE_LEN - 777, 777),
        (FILE_LEN - 1, 1),
        (FILE_LEN - 4096, 4096),
        (FILE_LEN - 4097, 4097),
        (FILE_LEN - 12288 - 5, 12288 + 5),
    ];
    for &chunk in &[0u64, 1, 4096, 5000, 8192, 12288, 64 * 1024] {
        for &(off, len) in cases {
            for align in [Align::End, Align::Start] {
                check(&f, off, len, chunk, align);
            }
        }
    }
    eprintln!("{} cases x 7 chunk sizes x 2 alignments clean", cases.len());
}

/// The same through an open handle, over both of its paths: the sector-aligned one that hands the
/// destination itself to the drive, and the bounce path.
#[test]
fn the_open_handle_against_the_guard_pages() {
    const FILE_LEN: u64 = 6 * MIB + 777;
    let f = Fixture::new("guard_handle", FILE_LEN);
    let h = unsafe { btb_open(f.wide.as_ptr()) };
    assert!(
        h >= 0,
        "btb_open -> {h} (os error {})",
        direct::last_os_error()
    );

    // the bounce path: an unaligned offset, an unaligned length, or both
    let bounce: &[(u64, u64)] = &[
        (0, 1),
        (1, 1),
        (0, 4095),
        (1, 4095),
        (4095, 1),
        (4095, 4098),
        (4096, 4097),
        (12287, 12289),
        (MIB + 7, 3 * MIB + 13),
        (777, 2 * MIB + 1),
        (FILE_LEN - 777, 777),
        (FILE_LEN - 1, 1),
        (FILE_LEN - 4097, 4097),
    ];
    // the aligned path: the sectors land in the fenced destination itself, the last one against the
    // guard page under Align::End (a Fence of a whole number of pages starts on a page either way)
    let aligned: &[(u64, u64)] = &[
        (0, 4096),
        (4096, 4096),
        (0, 2 * MIB),
        (MIB, 3 * MIB),
        (2 * MIB, 12288),
        (FILE_LEN - 777 - 4096, 4096),
    ];
    for &chunk in &[0u64, 1, 4096, 12288, 64 * 1024] {
        for &depth in &[1usize, 4, 16] {
            for &(off, len) in bounce.iter().chain(aligned) {
                for align in [Align::End, Align::Start] {
                    check_at(&f, h, off, len, chunk, depth, align);
                }
            }
        }
    }
    assert_eq!(btb_close(h), OK);
    eprintln!(
        "{} bounce + {} aligned cases x 5 chunk sizes x 3 depths x 2 alignments clean",
        bounce.len(),
        aligned.len()
    );
}

/// A read past the end of the file, or one whose range overflows, writes nothing outside the
/// destination and returns the code rather than a short buffer.
#[test]
fn refused_reads_leave_the_destination_alone() {
    const FILE_LEN: u64 = MIB + 5;
    let f = Fixture::new("guard_refused", FILE_LEN);
    let h = unsafe { btb_open(f.wide.as_ptr()) };
    assert!(h >= 0);
    for align in [Align::End, Align::Start] {
        let mut dst = Fence::<u8>::new("dst", 8192, align);
        dst.fill(0x3C);
        for &(off, len, want) in &[
            (FILE_LEN - 10, 100u64, btb_native::codes::ERR_EOF),
            (FILE_LEN, 1, btb_native::codes::ERR_EOF),
            (u64::MAX - 3, 8, btb_native::codes::ERR_DOMAIN),
        ] {
            let rc = unsafe { btb_read_direct(f.wide.as_ptr(), off, len, dst.mut_ptr(), 0) };
            assert_eq!(rc, want, "off={off} len={len}");
            let rc = unsafe { btb_read_at(h, off, len, dst.mut_ptr(), 0, 0) };
            assert_eq!(rc, want, "read_at off={off} len={len}");
            let rc = unsafe { btb_read_at(h + 1000, off, len, dst.mut_ptr(), 0, 0) };
            assert_eq!(rc, btb_native::codes::ERR_DOMAIN, "no such handle");
        }
        dst.check_borders("refused");
        // A short (EOF) read may have partially filled the destination; that is inside dst only.
        let mut dst2 = Fence::<u8>::new("dst2", 4096, align);
        dst2.fill(0x3C);
        let rc =
            unsafe { btb_read_direct(f.wide.as_ptr(), FILE_LEN - 4000, 4096, dst2.mut_ptr(), 0) };
        assert_eq!(rc, btb_native::codes::ERR_EOF);
        dst2.check_borders("refused-tail");
        let rc = unsafe { btb_read_at(h, FILE_LEN - 4000, 4096, dst2.mut_ptr(), 0, 0) };
        assert_eq!(rc, btb_native::codes::ERR_EOF);
        dst2.check_borders("refused-tail-at");
    }
    assert_eq!(btb_close(h), OK);
}

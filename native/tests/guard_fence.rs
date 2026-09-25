// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The fence itself: one element read past an `Align::End` payload, one read before an
//! `Align::Start` payload, and a write into a read-only payload each kill a child process with an
//! access fault, while the same buffers read cleanly in bounds. A fence that silently guards nothing
//! fails here instead of passing every other guard suite.

mod common;

use common::*;
use std::process::{Command, Output};

/// Set in a child to the probe it runs; unset in the parent that spawns them.
const PROBE_ENV: &str = "BTB_FENCE_PROBE";
/// Printed by a child right before its access, so a fault elsewhere in setup does not count.
const MARKER: &str = "fence probe: accessing";
/// A payload shorter than a page (poisoned slack beside it) and one of exactly 16 KiB, a whole
/// number of pages at 4 KiB and at Apple silicon's 16 KiB.
const LENS: [usize; 2] = [1001, 4096];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Probe {
    InBounds,
    PastEnd,
    BeforeStart,
    WriteReadOnly,
}

const PROBES: [Probe; 4] = [
    Probe::InBounds,
    Probe::PastEnd,
    Probe::BeforeStart,
    Probe::WriteReadOnly,
];

fn run_probe(probe: Probe, len: usize) {
    let align = if probe == Probe::BeforeStart {
        Align::Start
    } else {
        Align::End
    };
    let mut f = Fence::<u32>::new("probe", len, align);
    f.fill(7);
    f.protect_readonly();
    let p = f.mut_ptr();
    eprintln!("{MARKER} {probe:?} len {len}");
    unsafe {
        match probe {
            Probe::InBounds => {
                assert_eq!(std::ptr::read_volatile(p), 7);
                assert_eq!(std::ptr::read_volatile(p.add(len - 1)), 7);
            }
            Probe::PastEnd => {
                std::ptr::read_volatile(p.add(len));
            }
            Probe::BeforeStart => {
                std::ptr::read_volatile(p.sub(1));
            }
            Probe::WriteReadOnly => std::ptr::write_volatile(p.add(len / 2), 8),
        }
    }
    f.protect_readwrite();
    f.check_borders("in-bounds probe");
}

#[cfg(unix)]
fn faulted(out: &Output) -> bool {
    use std::os::unix::process::ExitStatusExt;
    matches!(out.status.signal(), Some(libc::SIGSEGV | libc::SIGBUS))
}

#[cfg(windows)]
fn faulted(out: &Output) -> bool {
    const STATUS_ACCESS_VIOLATION: u32 = 0xC000_0005;
    out.status.code() == Some(STATUS_ACCESS_VIOLATION as i32)
}

/// Each probe in a child of this same test binary: the out-of-bounds ones must die of an access
/// fault after reaching the access, the in-bounds one must pass.
#[test]
fn out_of_bounds_access_faults() {
    if let Ok(v) = std::env::var(PROBE_ENV) {
        let (name, len) = v.split_once(':').expect("probe:len");
        let probe = *PROBES
            .iter()
            .find(|p| format!("{p:?}") == name)
            .expect("known probe");
        run_probe(probe, len.parse().expect("len"));
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    for len in LENS {
        for probe in PROBES {
            let out = Command::new(&exe)
                .args([
                    "--exact",
                    "out_of_bounds_access_faults",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(PROBE_ENV, format!("{probe:?}:{len}"))
                .output()
                .expect("spawn the probe");
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            let ctx = format!("{probe:?} len {len}: {}\n{stdout}\n{stderr}", out.status);
            assert!(
                stderr.contains(MARKER),
                "{ctx}: the probe never reached its access"
            );
            if probe == Probe::InBounds {
                assert!(out.status.success(), "{ctx}: an in-bounds access failed");
                assert!(
                    stdout.contains("1 passed"),
                    "{ctx}: the probe test did not run"
                );
            } else {
                assert!(faulted(&out), "{ctx}: the access did not fault");
            }
        }
    }
    eprintln!(
        "{} probes x {} lengths: every out-of-bounds access faulted",
        PROBES.len(),
        LENS.len()
    );
}

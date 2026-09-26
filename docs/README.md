# btb docs

Documentation of implementation and behavior for various components within btb.

- **Guides**
  - [Tuning btb](./tuning.md)
    - Moving weights between VRAM/RAM/disk, getting the most tok/s, the speculative tree, greedy vs sampled decoding, and the `BTB_*` knobs.
  - [Building btb](./building.md)
    - Building from source: prerequisites (Rust, CUDA/`nvcc`, MLX), the dev checks and tests, and cross-building.
  - [Benchmarking](./benchmarking.md)
    - `btb bench`, the device/model matrix, and the `.venv-compare` setup with the rival engines.
- **Behavior**
  - [Bus Pass Scheduler](./disk-scheduler.md)
    - How a mixture-of-experts model streams experts from disk: the Route, the Bus Pass, and the Timetable.
  - [Sessions](./sessions.md)
    - A session's one state, the transactions every change goes through, and the leases forks and batches take.
  - [Speculative Decoding](#) - coming soon
- **Packed Stores**
  - [Pack-12](./pack-12.md)
    - For more information on packing models to lossless 12-bit, how it helps and what it costs.
  - [The MLX megakernel](./mega-mlx.md)
    - A dense Qwen3 decode pass as one Metal dispatch on Apple silicon: how, why, and when it applies.

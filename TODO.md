# TODO

## v0.1.0
- [x] Make the `devices` command prettier
- [x] support for piping the prompt into the CLI
- [x] remove the `ane` command in favor of the `pack-mlx` command which packs a model similar to `pack`, but instead uses `--mlx-mega`
- [x] --profile needs to become something that's easily sharable on github for bug reports, so we can diagnose the scheduler, inference, etc. in one fell swoop
- [x] tested and working certified CI wheel builders in github
- [x] Evan: write human docs
- [ ] Evan: Update the project description
- [ ] Evan: Un-slop all of the command descriptions, and all of the arguments for each command
- [ ] Evan: cancel sendgrid
- [x] --packed DIR should definitely not be its own arg, it should function the same as `path`
- [x] --verbose needs -v alias
- [x] --device needs -d alias
- [x] in serve, --host and --port need descriptions
- [x] instead of the dumbass "memory left for the rest of the machine:" as a section header, please do "scheduler:"
- [x] all fixtures need to be random for license purposes
- [x] unified script for generating a random fixture
- [x] Claude & Evan: Rewrite docs/disk-scheduler.md so its not paragraphs of narrative useless rambling, and instead describes what the feature is, how it works, and what it's for.
- [x] chat command should stream
- [x] openai server config? tool allow/limit params?
- [x] pi.dev support
- [x] top-p/top-k selection + fused kernel
- [x] redo the --profile so its not horrible for devs, instead - make the .md part of the crash detail output that can then be pasted into github issues
- [x] developer setup instructions
- [ ] interactive consent prompting for new folders/files (w/ --quiet/interactive session detection override)
- [x] see which env vars are worth dropping / unused
- [ ] hunt down every type that can be named with a more specific kinds.py equivalent
- [x] sys.path.insert cleanup in tests
- [ ] build an A2A data farming harness
- [ ] tighten up the benchmark splits for greedy / temp; make the 'descriptions' in the tables into hardcoded resident flags with pre-defined configuration

## v0.2.0
- [ ] multi-medium model support (image / vision / custom)
- [x] Sampling: `temperature` / `top_p` are ignored, every path is argmax. Add a sampler; speculation via rejection sampling.
- [ ] test on a 1GB system and see what explodes
- [ ] Support for multiple GPU/CPU devices
    - [ ] Support for multiple CPUs
- [ ] cache previously loaded MoE experts into some kind of witch soup that can be used to speed up inference?
- [ ] subcommand for `gguf`, that converts a model into `gguf` (convenience)
- [ ] actually give all those Json types true TypedDict definitions
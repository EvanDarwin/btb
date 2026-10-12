// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
// The roles of the card kernels' parameters. A role is a typedef of its element type (a pointer's: `const Query*
// __restrict__ q`) or of its type (a scalar's: `QHeads Hq`), named for what the value is; constness and
// __restrict__ stay at each parameter, and a typedef is the type it names, so a role changes no kernel's code.
// Every extern "C" kernel's parameters are roles, never bare types, and two parameters of one kernel never share one.
//
// `python build.py bindings` (native/cuda/bindings.py) reads this file and the kernels' signatures and writes each
// role as its own Python type (a NewType in btb/kinds.py: a card address for a pointer role, the value for a scalar)
// and a typed launcher per kernel (btb/engine/card_launch.py), so a launch with two arguments swapped, or a value of
// one role where another goes, fails the type check. A role is a pointer's everywhere or a scalar's everywhere.
// Each role's comment is its Python type's.

// --- a pass's rows ---
typedef int Rows;              // T: the rows a launch carries (a pass's tokens, a matvec's M)
typedef int PrefixLen;         // [1]: the cache's rows before the pass, read on the card at each replay
typedef int Depths;            // [T]: each row's depth in the pass's tree
typedef int TreeParents;       // [T]: each row's parent in the pass's tree, -1 the prefix, -2 padding; null a chain
typedef int RowsLayout;        // [3 + 3 T]: the rows as T sequences a token each (base, step, W, then col, off, len)
typedef int RowMap;            // [cap]: the cache's row map, a logical row to its slot in its page; null none

// --- matvecs and GEMMs ---
typedef bf16 Weight;           // [R, C] row-major: a matvec's or GEMM's weight
typedef bf16 MatIn;            // [T, C]: the rows a matvec or GEMM reads
typedef bf16 MatOut;           // [T, R]: the rows it writes
typedef float MatInF32;        // [T, C] float32: the host gemv's rows, read by its card twin
typedef float MatOutF32;       // [T, R] float32: the rows the host gemv's card twin writes
typedef int OutDim;            // R: the weight's rows, the output's width
typedef int InDim;             // C: the weight's columns, the input's width
typedef bf16 GateUp;           // [T, 2 I]: the MLP's gate then up projection
typedef bf16 MlpAct;           // [T, I]: act(gate) * up, the down projection's input
typedef int Intermediate;      // I: the MLP's intermediate width
typedef unsigned char Mx4Blocks;  // [seats, groups, 16]: MXFP4 experts' 32-value blocks, two fp4 codes a byte
typedef unsigned char Mx4Scales;  // [seats, groups]: each block's power-of-two scale, biased by 127
typedef int Seats;             // [n]: the experts' seats in a depot's stacks
typedef long long Mx4Groups;   // the 32-value blocks of one expert
typedef bf16 Widened;          // [n, groups * 32]: the experts widened to bf16
typedef int WarmShare;         // [1]: the share of each weight row warmed into L2, in 256ths, read on the card
typedef int WarmSink;          // [1]: where the warm-up's loads land, so they are kept
typedef int MmaMisses;         // [1]: the tensor cores' self-check's mismatches, zeroed before it

// --- norms, rope and the cache ---
typedef bf16 Residual;         // [T, H]: the residual stream, added to in place
typedef bf16 BlockOut;         // [T, H]: a block's output, added to the residual; null adds nothing
typedef bf16 NormWeight;       // [H]: an RMSNorm's weight
typedef float NormEps;         // an RMSNorm's epsilon
typedef int NormCentered;      // 1: the norm's weight is zero-centred (times 1 + w); 0: times w
typedef bf16 Normed;           // [T, H]: an RMSNorm's output
typedef int Hidden;            // H: the hidden width
typedef bf16 Qkv;              // [T, (Hq + 2 Hk) D]: the merged q, k and v projection's rows
typedef int QkvWidth;          // a Qkv row's width, in elements
typedef int QHeadStride;       // the elements between two query heads of a Qkv row
typedef int KColumn;           // where a Qkv row's keys start
typedef int VColumn;           // where a Qkv row's values start
typedef bf16 QNorm;            // [D]: the query heads' norm weight; null no norm
typedef bf16 KNorm;            // [D]: the key heads' norm weight; null no norm
typedef bf16 RopeCos;          // [positions, ts]: the model's rotary cosines
typedef bf16 RopeSin;          // [positions, ts]: the model's rotary sines
typedef int RopeStride;        // ts: a rotary table's row width
typedef int RopeDims;          // the dims a head's rope turns; the rest pass through
typedef int KeepRawKey;        // 1: the keys go into the cache as projected (the indexer's raw keys)
typedef bf16 Query;            // [T, Hq, D]: the query heads, normed and roped
typedef bf16 KCache;           // a layer's key rows: head g's row j at g * hs + j * rs
typedef bf16 VCache;           // a layer's value rows, laid out as its keys
typedef int QHeads;            // Hq: the query heads
typedef int KvHeads;           // Hk: the key and value heads
typedef int CacheHeadStride;   // hs: the elements between two heads of the cache
typedef int CacheRowStride;    // rs: the elements between two rows of the cache

// --- attention ---
typedef bf16 AttnOut;          // [T, Hq, D]: the attention's output
typedef float AttnScale;       // the scores' scale, 1 / sqrt(D) unless the model says otherwise
typedef int Window;            // a sliding layer's window, in keys; 0 the whole cache
typedef float PartMax;         // [S, T, Hq]: each split's running max, for the last block to fold
typedef float PartSum;         // [S, T, Hq]: each split's running sum of exponentials
typedef float PartAcc;         // [S, T, Hq, D]: each split's running output
typedef int Arrivals;          // [T, Hq]: the splits arrived so far, zero between launches
typedef int ChunkStart;        // n0: a prompt chunk's first position
typedef float PrefillRun;      // [T, Hq, D]: each row's running output across its key groups
typedef bf16 OutGate;          // the output gate's logits, at their own place in the query projection's rows
typedef int GateRowStride;     // the elements between two rows of the gate
typedef int GateHeadStride;    // the elements between two heads of the gate
typedef int HeadDim;           // D: a head's width
typedef bf16 GatedAttn;        // [T, Hq D]: the attention's output times sigmoid(gate)
typedef int AttnWidth;         // Hq D: an attention output row's width

// --- Qwen4's sparse attention's indexer ---
typedef bf16 IdxQuery;         // [T, Hi, DI]: the indexer's query heads
typedef bf16 IdxRawKey;        // [cap, DI]: the indexer's raw keys, a row a position
typedef bf16 IdxPooled;        // [NBcap, DI]: the committed complete blocks' pooled keys
typedef int IdxPooledLen;      // [2]: the pooled blocks, then the pool's arrival count
typedef bf16 IdxKNorm;         // [DI]: the indexer's key norm weight
typedef int IdxHeads;          // Hi: the indexer's heads
typedef int BlockRatio;        // r: the positions a block pools
typedef int TopBlocks;         // the blocks a row keeps
typedef int PooledCap;         // NBcap: the pooled keys' rows
typedef int MaxNewBlocks;      // the most blocks a pool adds
typedef float IdxScores;       // the indexer's block scores, its scratch
typedef int PickedBlocks;      // [T, ktop]: each row's picked blocks
typedef int PickedCount;       // [T]: each row's picked block count

// --- the step's handoff to the host ---
typedef long long StepToken;   // [1]: the step's token, on the card
typedef long long HostToken;   // pinned host memory: the step's token
typedef int HostLen;           // pinned host memory: the cache's length after the step
typedef long long HostClock;   // pinned host memory: the card's clock at the step's end, ns

// --- the gated DeltaNet ---
typedef float DeltaMixed;      // [T, C]: the q | k | v projection before its conv
typedef float DeltaZ;          // [T, Hv dv]: the output gate's input
typedef float DeltaA;          // [T, Hv]: the decay's input
typedef float DeltaB;          // [T, Hv]: the write strength's input
typedef bf16 DeltaProj;        // [T, ps]: the card program's merged projection, the nodes' inputs as its columns
typedef int ProjStride;        // ps: a DeltaProj row's width
typedef int QkvColumn;         // where a DeltaProj row's q | k | v starts
typedef int ZColumn;           // where a DeltaProj row's z starts
typedef int BColumn;           // where a DeltaProj row's b starts
typedef int AColumn;           // where a DeltaProj row's a starts
typedef int ProjRows;          // [T]: the DeltaProj row a node reads; null row j; negative ends the pass
typedef float ConvWeight;      // [C, K]: the conv's weight
typedef float ConvBias;        // [C]: the conv's bias; null none
typedef float ConvState;       // [C, K]: the conv's window
typedef float DeltaState;      // [Hv, dk, dv]: the recurrent state
typedef float DeltaScratch;    // [slots, Hv, dk, dv]: the tree's node states; null steps the state in place
typedef int ScratchSlots;      // [T]: the scratch slot a node's state goes to; null a slot a node
typedef float DeltaALog;       // [Hv]: the decay's log rate
typedef float DeltaDtBias;     // [Hv]: the decay's time-step bias
typedef float DeltaNormWeight; // [dv]: the output norm's weight
typedef int DeltaGate;         // the output gate: 1 sigmoid, else silu
typedef int ConvChannels;      // C: a conv's channels
typedef int ConvTaps;          // K: a conv's taps
typedef int ConvDilation;      // a conv's dilation
typedef int DeltaKHeads;       // the DeltaNet's key heads
typedef int DeltaVHeads;       // Hv: the DeltaNet's value heads
typedef int DeltaKDim;         // dk: a key head's width
typedef int DeltaVDim;         // dv: a value head's width
typedef float DeltaOutF32;     // [T, Hv dv] float32: the DeltaNet's output
typedef bf16 DeltaOut;         // [T, Hv dv]: the DeltaNet's output, rounded to bf16

// --- Qwen4's hyper-connection streams ---
typedef bf16 Streams;          // [T, G, H]: the streams, added to in place
typedef bf16 Inject;           // [T, G]: the block's inject weights
typedef int StreamCount;       // G: the streams
typedef bf16 MixDown;          // [T, ds]: the mixer's down projection rows (then the inject logits)
typedef int MixDownStride;     // ds: a MixDown row's width
typedef bf16 MixAct;           // [T, R]: silu(down / G), the mixer's low-rank activation
typedef int MixRank;           // R: the mixer's rank
typedef bf16 MixUp;            // [T, G, H]: the mixer's up projection's logits
typedef bf16 Mixed;            // [T, H]: the streams mixed

// --- Qwen4's MoE ---
typedef bf16 RouterLogits;     // [T, ls]: the experts' logits, then the shared expert's gate logit
typedef int RouterStride;      // ls: a RouterLogits row's width
typedef int Experts;           // E: the routed experts
typedef int TopExperts;        // k: the experts a row takes
typedef int ExpertIds;         // [T, k]: each row's experts
typedef bf16 ExpertWeights;    // [T, k]: each row's expert weights
typedef int HostExpertIds;     // pinned host memory: each row's experts; null none
typedef bf16 HostExpertWeights;  // pinned host memory: each row's expert weights
typedef unsigned RouteCount;   // [1]: the router's blocks arrived, zero between launches
typedef int RouteSeq;          // [1]: the routing's sequence number, on the card
typedef int HostRouteSeq;      // pinned host memory: the sequence number of the last rows out; null none
typedef bf16 Routed;           // [T, H]: the routed experts' sum
typedef bf16 SharedOut;        // [T, H]: the shared expert's output
typedef bf16 SharedGate;       // the shared expert's gate logit, row t at t * ls
typedef bf16 MoeOut;           // [T, H]: the MoE's output; may be Routed's buffer

// --- Qwen4's per-layer n-gram embedding ---
typedef bf16 PleKv;            // [T, ks]: the n-gram embeddings' key then value projection
typedef int PleKvStride;       // ks: a PleKv row's width
typedef bf16 PleQuery;         // [T, G H]: the streams through the query norm
typedef bf16 PleKNorm;         // [G H]: the key norm's weight
typedef bf16 PleConvNorm;      // [G H]: the conv norm's weight
typedef float InvSqrtHidden;   // 1 / sqrt(H)
typedef bf16 PleGated;         // [T, G H]: sigmoid(gate) * value
typedef bf16 PleNormed;        // [T, G H]: the gated rows through the conv norm
typedef bf16 PleConvWeight;    // [C, K]: the dilated conv's weight
typedef bf16 PleWindow;        // [C, (K - 1) dil]: the conv's kept window, oldest first
typedef int PleUpdate;         // 1: a one-row step shifts its row into the window

// --- sampling ---
typedef float Logits;          // [R, V] float32: the rows' logits
typedef int Vocab;             // V: a Logits row's width
typedef float SampleParams;    // [2]: 1 / temperature, top_p
typedef unsigned SampleMax;    // [R]: each row's max scaled logit, as ordered bits
typedef int RadixLevel;        // the radix level, 0 to 2
typedef int SampleMode;        // 0 counts (top-k), 1 masses (top-p)
typedef unsigned RadixPrefix;  // [R]: the threshold resolved so far, as ordered bits
typedef unsigned RadixFloor;   // [R]: the floor a token must clear, as ordered bits
typedef unsigned long long RadixHist;   // [R, 2048]: a level's histogram
typedef unsigned long long RadixCarry;  // [R]: the remainder a level carries to the next
typedef int TopK;              // the tokens top-k keeps; 0 all
typedef unsigned TopKFloor;    // [R]: the top-k threshold, as ordered bits
typedef unsigned TopPFloor;    // [R]: the top-p threshold, as ordered bits
typedef unsigned long long SampleBest;  // [R]: each row's best draw, packed
typedef unsigned SampledTokens;  // [R]: each row's token
typedef unsigned NoiseSeed;    // [2]: the seed, low word first
typedef int NoisePos;          // [1]: the position the noise is keyed to
typedef int NoiseSalt;         // the key's salt; 0 none
typedef unsigned NoiseKeys;    // [R, 2]: each row's noise key, low word first
typedef float DraftProbs;      // [T, Vd]: the drafter's distribution a node
typedef int DraftKids;         // [T, C]: each node's drafted children in draw order, -1 padding
typedef unsigned HasDraftProbs;  // [T]: 0 where the drafts are point masses
typedef int DraftVocab;        // Vd: a DraftProbs row's width
typedef int DraftWidth;        // C: a node's drafted children
typedef unsigned VerifyPicks;  // [T]: each node's pick, (slot + 1) << 24 | token

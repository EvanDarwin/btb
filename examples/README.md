# Examples

Examples of using `btb` via its Python interface.

| Script               | Description                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------- |
| `ask.py`             | load, one answer, one streamed                                                                        |
| `chat.py`            | a conversation: the cache reused turn to turn                                                         |
| `plan_first.py`      | `btb.plan`: the placement priced before loading; a machine that cannot carry it refused by name       |
| `memory_budget.py`   | the host budget, the reserves, the ledger, the grant gate refusing by name                            |
| `coexist.py`         | your own tensors beside the engine: grant and reserve through the same ledger                         |
| `shed_and_regrow.py` | the policies' moves by hand: a layer shed to the drive or off the card and back, the answer unchanged |
| `custom_loop.py`     | your own tokens and your own loop over `forward` and the cache                                        |
| `batch.py`           | many prompts at once, the epochs sized by the scheduler                                               |
| `spans.py`           | drafts from text you hold, verified to the greedy answer                                              |
| `packed_store.py`    | the 12-bit store: pack a model, load from the store, the same tokens                                  |
| `foreign_model.py`   | the scheduler over a transformers model that is not btb's                                             |
| `openai_server.py`   | the OpenAI server in your own process                                                                 |
| `host_monitor.py`    | the torch-free machine readers under the memory system                                                |

# Composite traffic shaper

Network Modeling Homework 2: an empirical ON-OFF packet source from Homework 1 followed by a token-bucket filter (TBF) and leaky-bucket filter (LBF). The simulator uses logical time, packet descriptors, finite FIFO buffers, and whole-packet transmission.

The workload was fitted to receiver-observed IP packets during one prerecorded YouTube session. It is an open-loop synthetic source: there is no TCP congestion-control, retransmission-recovery, or video-player feedback model. Packet lengths include IP headers and retained retransmissions, so the measured byte rate is not application goodput.

## Install and test

Use Python 3.12. Runtime dependencies are pinned in `requirements.txt`: NumPy 2.4.3, SciPy 1.15.2, and Matplotlib 3.9.3. From this repository's root:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

For runtime use without tests, install `requirements.txt` instead. No packet capture, external service, or real-time waiting is needed. The optional source-equivalence integration test runs only when the original HW1 workspace is available; portable regression tests remain self-contained.

## Reproduce the study

```sh
python experiments.py
python make_figures.py
```

The first command fits the source, calibrates workload scales, screens the parameter grid, refines the output-rate boundary, evaluates a shortlist on independent traces, confirms fixed candidates over longer horizons, and runs the small Poisson control. It writes stage summaries under `data/results/`, selected shaper parameters under `config/`, and the required trace pair under `data/traces/`.

The full protocol and seed ranges are specified in `experiments.py`; generated `config/experiment.json` records the settings used. It is a record of the study, not a configuration input that changes the runner. The five shaper parameters can be changed independently through the standalone Shaper CLI below.

After a completed study, figures can be regenerated directly from the saved results without rerunning simulations:

```sh
python make_figures.py
```

Figures and numerical curve exports are written to the ignored `figure/` directory. To reuse already saved stages from the **same unchanged study**:

```sh
python experiments.py --resume
```

Resume trusts existing stage files; it is not a general cache-invalidation system. Run afresh after changing fitting inputs, source behavior, simulator semantics, or the experimental protocol. Within each phase, identical generated inputs are reused across configurations. Screening, evaluation, horizon confirmation, and controls use separate seed ranges.

## Run the Source and Shaper separately

After the study has produced `config/selected.json`, these commands recreate its required representative trace pair using the first evaluation seed:

```sh
python source.py --model onoff --duration 3600 --seed 2026111500 --output data/traces/Input_Trace_File.txt
python shaper.py --input data/traces/Input_Trace_File.txt --config config/selected.json --output data/traces/Output_Trace_File.txt
```

The Source also accepts `--fitting PATH` and a Poisson control through `--model poisson --rate BYTES_PER_SECOND`. That rate is an expected byte rate; individual realizations need not have the same packet count or bytes as ON-OFF realizations.

The Shaper accepts a JSON object with these five keys, either directly or inside a `parameters` object:

| Key | Unit and meaning |
| --- | --- |
| `rho` | Tokens/s; one token permits one byte |
| `b` | Token capacity; must accommodate the largest input packet |
| `Bi` | Input waiting capacity in bytes |
| `mu` | LBF transmission rate in bytes/s |
| `Bo` | Output waiting capacity in bytes |

Both trace files are headerless UTF-8 text, with one row per packet: integer identifier, timestamp in seconds, integer length in bytes. Input times are generation times. Output times are transmission-completion times, and discarded packets are absent. Identifiers and lengths are preserved; equal input timestamps retain file order. Times are written with round-trip precision.

The assignment's main text uses `Input_Traffic_Trace.txt` / `Output_Traffic_Trace.txt`, whereas its Deliverables section uses `Input_Trace_File.txt` / `Output_Trace_File.txt`. This repository follows the latter names. The CLIs accept arbitrary file paths.

## Simulation and measurement conventions

- Queues initially are empty and the token bucket is full. Tokens accumulate continuously up to `b`; TBF release is instantaneous and consumes a whole packet's credit.
- `Bi` counts all packets waiting for tokens. `Bo` counts packets waiting for the LBF and excludes the packet currently in transmission. Zero waiting capacity still permits immediate forwarding or service when possible.
- Service is FIFO and non-preemptive, taking `length / mu` seconds. At tied times, output completions and queued work advance before new arrivals. New packets cannot overtake existing input waiters.
- A packet that does not fit is dropped at that buffer. Output drops consume their TBF tokens; there is no token refund or downstream backpressure.
- Generation stops at the trial horizon, then accepted packets drain to completion. Queued packets at the horizon are not treated as losses.

The assumed design budgets are packet loss at most 0.1%, and lost-or-late fraction at most 5% for a five-second added-shaping-delay deadline. These are engineering assumptions for this study, not official YouTube requirements. A deadline failure is either a dropped packet or one completed more than the allowed time after generation; a completion exactly at its deadline is on time. One- and ten-second deadline fractions are also retained.

Loss and deadline fractions use all generated packets as their denominator. Delay summaries use delivered packets. Throughput is delivered IP bytes divided by the later of the generation horizon and last completion time, so it includes draining. Each independent run supplies one observation for the reported means and nominal Student-t intervals. Zero observed failures do not certify zero failure probability, and simulation intervals do not cover uncertainty in the nine-cycle empirical source fit.

## Files and provenance

- `source.py`, `shaper.py`, and `metrics.py`: source generation, simulation, and measurement.
- `experiments.py` and `make_figures.py`: the experiment protocol and figure reproduction.
- `data/fitting/packets.csv`: the 32,768 HW1 fitting packets from capture time `[60,270)` seconds; the holdout is excluded.
- `data/provenance.json`: original data location and hashes, selection rules, limitations, and assistance information.
- `data/results/`: per-run and aggregate numerical results, with representative diagnostic arrays.
- `data/traces/`: one complete matched input/output pair, using seed 2026111500.
- `config/`: the selected parameters and experimental settings.

The report, LaTeX bundle, learning notes, original assignment, and course materials are maintained separately from this code/data repository.

## Saved study outcome

The bounded search tested 251 configurations and evaluated all eight screening-feasible candidates independently. The selected configuration is R001: output rate 10.471144 Mbit/s, token rate 350596.337083 tokens/s, bucket 5250385 tokens, input waiting memory 5250384 bytes, and output waiting memory 5250385 bytes.

Across twenty one-hour evaluation runs, mean loss was 0.016934% and mean lost-or-late fraction at five seconds was 2.931280%. Ten fresh two-hour confirmation runs gave 0.002839% and 2.628871%. These meet the assumed targets for average run fractions, not every run; the worst evaluation run exceeded both budgets. Rare-loss samples are zero-heavy, so nominal intervals are not probability guarantees. See the complete per-run evidence in data/results.

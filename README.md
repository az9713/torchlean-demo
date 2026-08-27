<h1 align="center">
  <img src="home_page/assets/media/brand/torchlean-logo.png" alt="TorchLean logo" width="88" align="center">
  Formalizing Neural Networks in Lean
</h1>

TorchLean brings neural-network programming and formal reasoning into one Lean project. Tensor
shapes are part of the types, models are executable Lean programs, and the same definitions can be
used by training code, graph transformations, certificate checkers, and proofs. CPU and CUDA
backends handle numerical work; the Lean library records the mathematical meaning and assumptions
attached to each path.

## Test suite manual — live page

[![TorchLean Test Suite Manual](torchlean-tests-guide.png)](https://az9713.github.io/torchlean-demo/torchlean-tests-guide.html)

`run_tests.sh` is a 12-case smoke suite for a real install: 11 CPU cases plus one opt-in CUDA case.
Click the screenshot above for the **[live manual](https://az9713.github.io/torchlean-demo/torchlean-tests-guide.html)**,
which explains what each case proves, how to start it from Git Bash on Windows, and how to read PASS/FAIL.

```bash
./run_tests.sh          # CPU cases only
./run_tests.sh --cuda   # adds the GPU case
```

## Installation

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake exe cache get
lake build
```

For Linux, macOS, Windows/WSL, CUDA, optional LibTorch support, and an explanation of
TorchLean's backend architecture, see the [Installation guide](https://lean-dojo.github.io/TorchLean/installation/).

TorchLean is pinned by `lean-toolchain` and currently builds with
`leanprover/lean4:v4.33.0`.

## Quickstart

```bash
lake exe torchlean quickstart_mlp --device cpu --steps 10 --scalar ieee32-exec --execution eager
lake exe torchlean quickstart_mlp --device cpu --steps 10 --scalar float32 --execution eager

# Optional CUDA run, if the CUDA toolkit and an NVIDIA GPU are available:
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 1000
```

The first quickstart uses TorchLean's independent raw-bit binary32 reference. The second uses
Lean's native `Float32` arithmetic. The CUDA command selects the native GPU runtime and reports an
error when CUDA is unavailable.

Application code looks like this:

```lean
import NN.API
open TorchLean

/-- A two-layer regression model. The dimensions are checked when the layers are composed. -/
def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

-- Four input rows, each containing two features.
def xs : Tensor Float (shape![4, 2]) :=
  tensorOfList! [4, 2] [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]

-- One regression target for each input row.
def ys : Tensor Float (shape![4, 1]) :=
  tensorOfList! [4, 1] [0.2, 1.0, 1.0, 1.8]

-- The dataset type records the feature and target shapes expected by the trainer.
def data : Trainer.DataSource (.dim 2 .scalar) (.dim 1 .scalar) :=
  Data.tensorDataset xs ys

def trainOnce : IO Unit := do
  -- Select the loss and train through a typed graph interpreted by IEEE32Exec.
  let trainer :=
    Trainer.new model
      { task := .regression
        optimizer := optim.sgd { lr := 0.05 }
        execution := .typedGraph
        device := .cpu
        scalar := .ieee32Exec }
  -- Inspect the initialized model before any parameter updates.
  let initialPrediction ← trainer.predict (tensorOfList! [2] [0.5, -0.25])
  IO.println s!"initial={Tensor.pretty initialPrediction}"
  -- Each step averages 16 sample gradients at one parameter point, then updates once.
  -- Training returns a new trainer containing the updated parameters and run history.
  let trained ← trainer.train data { steps := 200, batchSize := 16, logEvery := 25 }
  trained.printSummary
```

## Commands

```bash
lake exe torchlean --help
lake exe verify --help
lake exe verify -- torchlean-ibp
```

For the maintained examples:

```bash
lake build NNExamples
```

## Use TorchLean From Another Lean Project

TorchLean is a normal Lake package. You can depend on the Git repository directly:

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

Then run:

```bash
lake update
lake exe cache get
lake build
```

Use `import NN.API` for model, data, and training code. It provides `TorchLean.nn`,
`TorchLean.Data`, `TorchLean.Trainer`, and `TorchLean.optim`, together with the mathematical model
definitions in `Spec`. Direct verifier lowering has the focused import `NN.API.Verification`. Use
`import NN` when the same file also needs proofs or backend infrastructure; focused imports such
as `NN.GraphSpec`, `NN.Runtime`, or `NN.Proofs` are available for subsystem work.

Downstream model and training files should start from:

```lean
import NN.API
open TorchLean
```

The floating-point library can also be used on its own:

```lean
import NN.Floats
open TorchLean.Floats
```

This import provides generic formats and rounding, finite binary32 semantics, executable IEEE
binary32 operations, interval rounders, and scalar quantization. It does not import tensors,
models, autograd, CUDA, certificate checkers, or external numerical tools. More specialized users
can import `NN.Floats.NeuralFloat`, `NN.Floats.FP32`, `NN.Floats.IEEEExec`, or
`NN.Floats.Interval` directly. Tensor quantization and runtime-approximation proofs are separate
adapters under `NN.Spec.Quantization` and `NN.Proofs.RuntimeApprox.FP32`.

For local development against a checkout, use a path dependency instead:

```lean
require TorchLean from "../TorchLean"
```

## Repository Map

- `NN.lean`: complete import for model, tensor, data, training, verification, and proof workflows.
- `NN/API`: the application API exported by `import NN.API` and included by `import NN`.
- `NN/Spec`: mathematical tensor, layer, model, and dynamical-system definitions.
- `NN/Runtime`: executable autograd, optimizers, training loops, CUDA boundary,
  PyTorch import/export, and RL runtime support.
- `NN/Backend`: contract-carrying backend capsules, profiles, device targets, reports, and gates.
- `NN/IR` and `NN/GraphSpec`: graph IR, graph semantics, and typed architecture
  descriptions.
- `NN/Proofs`: tensor algebra, selected autograd correctness theorems, analytic derivatives,
  runtime approximation, and bridge proofs.
- `NN/Floats`: finite-precision models, IEEE-style executable semantics,
  NeuralFloat formats, and error-bound infrastructure.
- `NN/MLTheory`: learning theory, robustness, CROWN/LiRPA, generative objectives,
  optimization theory, and related proof layers.
- `NN/Verification`: certificate checkers and CLI workflows.
- `NN/Examples`: quickstarts, model zoo commands, widgets, bundled verification assets,
  and interoperability workflows.
- `blueprint/TorchLeanBlueprint/Guide`: source for the guide.
- `home_page`: project website sources.

## Proofs And Runtime Boundaries

TorchLean proves properties of explicit Lean definitions. It also checks certificates produced by
external tools, including bound-propagation and scientific-computing workflows. A successful
certificate check proves the predicate implemented by that checker; it does not certify the program
that produced the certificate.

CPU instructions, CUDA kernels, cuBLAS, LibTorch, PyTorch, Julia, and other external systems are
runtime providers. Their interfaces, assumptions, and available checks are listed in
[`TRUST_BOUNDARIES.md`](TRUST_BOUNDARIES.md). Third-party sources and licenses are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and [`AI_USAGE.md`](AI_USAGE.md) describes the
project's use of coding assistants.

## Citation

If TorchLean is useful in your work, please cite
[*TorchLean: Formalizing Neural Networks in Lean*](https://arxiv.org/abs/2602.22631):

```bibtex
@misc{george2026torchlean,
  title         = {TorchLean: Formalizing Neural Networks in Lean},
  author        = {George, Robert Joseph and Cruden, Jennifer and Adkisson, Will and
                   Zhong, Xiangru and Zhang, Huan and Anandkumar, Anima},
  year          = {2026},
  eprint        = {2602.22631},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MS},
  url           = {https://arxiv.org/abs/2602.22631}
}
```

## License

TorchLean is released under the MIT License. See `LICENSE`.

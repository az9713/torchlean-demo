#!/usr/bin/env bash
# TorchLean smoke suite. Start it from Git Bash on Windows; it runs itself inside WSL.
#   ./run_tests.sh            CPU cases only
#   ./run_tests.sh --cuda     adds the GPU case (forces a separate build, slow first time)
# Env: TL_REPO=<WSL path to the checkout>   (default ~/TorchLean; give a WSL path, not C:\...)

# --- Git Bash -> WSL relaunch ------------------------------------------------
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    win="$(cd "$(dirname "$0")" && pwd -W)/$(basename "$0")"
    case "$win" in *\ *) echo "Path has a space; move the folder: $win"; exit 2;; esac
    # ponytail: expand everything here, in Git Bash. A '$' inside the wsl.exe
    # payload arrives EMPTY, so the payload must reach WSL as literal text.
    pre=""; [ -n "$TL_REPO" ] && pre="TL_REPO=$TL_REPO "
    export MSYS_NO_PATHCONV=1   # else Git Bash rewrites /home/... into C:/Program Files/Git/home/...
    exec wsl.exe -d Ubuntu -- bash -lc "${pre}bash $(echo "$win" | sed 's|^\(.\):|/mnt/\l\1|') $*"
    ;;
esac

# --- inside WSL from here ----------------------------------------------------
export PATH="$HOME/.elan/bin:$PATH"
REPO="${TL_REPO:-$HOME/TorchLean}"
CUDA=0
[ "$1" = "--cuda" ] && CUDA=1
cd "$REPO" || { echo "no checkout at $REPO"; exit 2; }

pass=0; fail=0; out=""
tl() { timeout 3600 "$@" 2>&1; }               # output is the evidence, not the exit code
say() { # name  ok(0=pass)  evidence
  if [ "$2" = 0 ]; then printf 'PASS  %-22s %s\n' "$1" "$3"; pass=$((pass+1))
  else printf 'FAIL  %-22s %s\n' "$1" "$3"; fail=$((fail+1)); echo "$out" | tail -12 | sed 's/^/        | /'
  fi
}
has() { echo "$out" | grep -qF "$1"; }
loss() { echo "$out" | grep -o "$1=[0-9.]*" | tail -1 | cut -d= -f2; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a<b && b!="")}'; }

echo "TorchLean suite — $REPO @ $(git rev-parse --short HEAD 2>/dev/null)"
echo "First run of a case may compile Lean modules (minutes). Later runs are seconds."
echo

# 01 typed tensors: the proof-level element type must refuse to print
out=$(tl lake exe torchlean quickstart_tensors)
if has '[IEEE32Exec]' && has 'Refusing to print'; then
  say 01-tensor-types 0 "Float/ℚ/Int/Float32/IEEE32Exec printed; Tensor ℝ refused"
else say 01-tensor-types 1 "missing IEEE32Exec row or the Tensor ℝ refusal"; fi

# 02 autograd against a closed form: d/dx mean(x^2) = x/n for x=[1.0,-2.4]
out=$(tl lake exe torchlean quickstart_autograd)
if has 'grad1(mean(x^2)) = [0.500000, -1.200000]'; then
  say 02-autograd-closed 0 "grad1(mean(x^2)) = [0.500000, -1.200000] exact"
else say 02-autograd-closed 1 "gradient differs from the closed form"; fi

# 03 training through the typed-graph executor: the loss must fall
out=$(tl lake exe torchlean quickstart_mlp --steps 200 --scalar ieee32-exec --execution typed-graph)
a=$(loss loss0); b=$(loss loss1)
if lt "$b" "$a"; then say 03-train-typedgraph 0 "loss $a -> $b (falls)"
else say 03-train-typedgraph 1 "loss $a -> $b (did not fall)"; fi

# 04 real-data training + the artifact it must leave behind
rm -f data/model_zoo/mlp_trainlog.json
out=$(tl lake exe torchlean mlp --steps 50)
a=$(loss loss0); b=$(loss loss1)
if lt "$b" "$a" && [ -s data/model_zoo/mlp_trainlog.json ]; then
  say 04-train-autompg 0 "loss $a -> $b, trainlog JSON written"
else say 04-train-autompg 1 "loss $a -> $b, trainlog missing or empty"; fi

# 05 model -> IR lowering -> interval bounds
out=$(tl lake exe verify -- torchlean-ibp)
n=$(echo "$out" | grep -o 'lowered IR nodes: [0-9]*' | grep -o '[0-9]*')
lo=$(echo "$out" | grep 'output box lo' | grep -o '[0-9.]*'); hi=$(echo "$out" | grep 'output box hi' | grep -o '[0-9.]*')
if [ "$n" = 18 ] && lt "$lo" "$hi"; then say 05-ibp-lowering 0 "18 IR nodes, box [$lo, $hi]"
else say 05-ibp-lowering 1 "nodes=$n box=[$lo, $hi] (expected 18 nodes, lo<hi)"; fi

# 06 a good external certificate is accepted
out=$(tl lake exe verify -- lirpa-mlp)
if has 'IBP certificate verified'; then say 06-cert-accept 0 "serialized bounds enclose the Lean recomputation"
else say 06-cert-accept 1 "the bundled mlp_cert.json was not accepted"; fi

# 07 robustness counts from the paper: certified must not exceed nominal
out=$(tl lake exe verify -- margin-cert)
e=$(echo "$out" | grep -o 'examples=[0-9]*' | cut -d= -f2)
nom=$(echo "$out" | grep -o 'nominal_ok=[0-9]*' | cut -d= -f2)
cer=$(echo "$out" | grep -o 'certified_ok=[0-9]*' | cut -d= -f2)
if [ "$e/$nom/$cer" = "360/349/318" ]; then say 07-margin-cert 0 "$nom/$e nominal, $cer/$e certified"
else say 07-margin-cert 1 "got $e/$nom/$cer, expected 360/349/318"; fi

# 08 structural check of an alpha,beta-CROWN leaf artifact
out=$(tl lake exe verify -- abcrown-leaf)
if has 'ok=1, bad=0'; then say 08-abcrown-leaf 0 "1 leaf checked, ok=1 bad=0"
else say 08-abcrown-leaf 1 "leaf artifact check did not report ok=1 bad=0"; fi

# 09 NEGATIVE: bounds moved by +1000 must be rejected, not waved through
python3 - <<'PY'
import json
d=json.load(open('NN/Examples/Verification/LiRPA/mlp_cert.json'))
d['result']['lo']=[x+1000.0 for x in d['result']['lo']]
json.dump(d,open('/tmp/tl_tampered_cert.json','w'))
PY
out=$(tl lake exe verify -- lirpa-mlp /tmp/tl_tampered_cert.json)
if has 'IBP certificate mismatch'; then say 09-neg-tampered 0 "tampered bounds rejected (mismatch reported)"
else say 09-neg-tampered 1 "tampered certificate was NOT rejected"; fi

# 10 NEGATIVE: a truncated artifact must fail parsing, not half-load
head -c 120 NN/Examples/Verification/LiRPA/mlp_cert.json > /tmp/tl_malformed_cert.json
out=$(tl lake exe verify -- lirpa-mlp /tmp/tl_malformed_cert.json)
if has 'invalid JSON'; then say 10-neg-malformed 0 "truncated artifact rejected at parse time"
else say 10-neg-malformed 1 "malformed artifact was NOT rejected"; fi

# 11 KNOWN FAILURE upstream: the bundled PINN fixture disagrees with Lean.
#    This passes while the bug is present; it flips to FAIL when upstream fixes it.
out=$(tl lake exe verify -- pinn-cert)
if has 'u(x-h) mismatch at x=0.250000'; then say 11-xfail-pinn-cert 0 "known: bundled pinn_cert.json still mismatches"
else say 11-xfail-pinn-cert 1 "pinn-cert changed — re-check this case, upstream may be fixed"; fi

# 12 GPU, opt-in
if [ "$CUDA" = 1 ]; then
  out=$(tl lake -R -K cuda=true exe torchlean mlp --device cuda --steps 200)
  a=$(loss loss0); b=$(loss loss1)
  if has 'device: cuda' && lt "$b" "$a"; then say 12-cuda-train 0 "on cuda, loss $a -> $b"
  else say 12-cuda-train 1 "loss $a -> $b (check device line and CUDA toolkit 13.1)"; fi
fi

echo
echo "$pass passed, $fail failed"
[ "$fail" = 0 ]

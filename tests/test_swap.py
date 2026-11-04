"""Tests for experiments/E5_1_swap. Run with `pytest tests/test_swap.py` or
`python tests/test_swap.py`. The __main__ runner needs no pytest."""
import contextlib
import copy
import importlib.util
import io
import math
import sys
import tempfile
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e5 = load("e5_run", ROOT / "experiments" / "E5_1_swap" / "run.py")
an = load("e5_analyze", ROOT / "experiments" / "E5_1_swap" / "analyze.py")
RecallModel, MultiHeadAttention = e5.RecallModel, e5.e11.MultiHeadAttention

CFG = dict(e5.SMOKE, batch=16, eval_batch=16, steps=3, warmup=1, refit_steps=5, refit_warmup=2)
NOISE, SEED, N = 0.6, 7, 32
PAIRS = [(t, i) for t in e5.ARMS for i in e5.readouts_for(t)]


def trained(arm):
    return e5.train_model(CFG, arm, SEED, 3e-3, NOISE)[0]


def batch(n=N):
    return e5.make_batch(e5.data_gen("test", SEED, NOISE, n, CFG), CFG["eval_batch"], n, NOISE, CFG)


@torch.no_grad()
def output(model, data):
    return model(data[0], data[1], data[2])


@torch.no_grad()
def cross_scores(model, data):
    model.ca.capture = True
    model(data[0], data[1], data[2])
    z = model.ca.last[0]
    model.ca.capture, model.ca.last = False, None
    return z


def fresh_model(readout):
    """Constructed with the readout, with init weights that differ from any trained model."""
    torch.manual_seed(12345)
    sa_name, ca_name = e5.READOUTS[readout]
    model = RecallModel(CFG, sa_name)
    if ca_name != sa_name:
        model.ca = MultiHeadAttention(CFG["d_model"], CFG["n_heads"], normalizer=ca_name, train_n=CFG["train_n"],
                                      kappa=CFG["kappa"], calib_p=CFG["calib_p"], calib_iters=CFG["calib_iters"])
    return model


def test_swap_equals_fresh_module_with_same_weights():
    data = batch()
    for t, i in PAIRS:
        swapped = e5.view(trained(t), i, CFG)
        fresh = fresh_model(i)
        assert set(swapped.state_dict()) == set(fresh.state_dict()), (t, i)
        fresh.load_state_dict(swapped.state_dict())  # strict
        assert torch.equal(output(swapped, data), output(fresh, data)), (t, i)


def test_rem_buffer_appears_and_goes():
    soft = trained("softmax")
    assert not hasattr(soft.ca, "kappa")
    rem = e5.view(soft, "rem", CFG)
    assert rem.ca.kappa.shape == (CFG["n_heads"], 1, 1) and rem.sa.kappa.shape == (CFG["n_heads"], 1, 1)
    assert torch.equal(rem.ca.kappa, RecallModel(CFG, "rem").ca.kappa)
    back = e5.view(rem, "softmax", CFG)
    assert "ca.kappa" not in back.state_dict() and "sa.kappa" not in back.state_dict()
    assert not hasattr(soft.ca, "kappa"), "the original model must not change"


def test_readouts_really_differ():
    data, soft = batch(), trained("softmax")
    outs = {i: output(e5.view(soft, i, CFG), data) for i in e5.READOUTS}
    names = list(outs)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            assert not torch.equal(outs[names[a]], outs[names[b]]), (names[a], names[b])


def test_cross_scores_depend_on_self_normalizer_only():
    """Why retrieval pins the self normalizer: the pre-registration's claim holds for the cross
    normalizer only. If the second assert ever fails, the pin has become unnecessary."""
    data, soft = batch(), trained("softmax")
    base = cross_scores(soft, data)
    assert torch.equal(base, cross_scores(e5.view(soft, "entmax15_cross", CFG), data))
    assert not torch.equal(base, cross_scores(e5.view(soft, "entmax15", CFG), data))


def test_retrieval_does_not_change_with_readout():
    for t in e5.ARMS:
        model = trained(t)
        base = e5.retrieval(model, t, CFG, "test", SEED, NOISE, N, 2)
        assert {"ret_head0", "ret_head1", "ret_best", "matched_filter_acc"} == set(base)
        for i in e5.readouts_for(t):
            assert e5.retrieval(e5.view(model, i, CFG), t, CFG, "test", SEED, NOISE, N, 2) == base, (t, i)
        own = e5.eval_pass(e5.view(model, t, CFG), CFG, "test", SEED, NOISE, N, 2)
        assert own["ret_best"] == base["ret_best"]


def test_retrieval_counts_by_hand():
    model, data = trained("entmax15"), batch()
    res = e5.eval_pass(model, CFG, "test", SEED, NOISE, N, 1)
    z, idx = cross_scores(model, data), data[4]
    hit = z[:, :, 0].argmax(-1) == idx[:, None]
    assert res["ret_head0"] == e5.clean(hit[:, 0].float().mean())
    assert res["ret_head1"] == e5.clean(hit[:, 1].float().mean())
    assert res["ret_best"] == e5.clean(hit.any(1).float().mean())
    assert res["ret_best"] >= max(res["ret_head0"], res["ret_head1"])


def test_refit_parameter_sets_are_exact():
    names = {n for n, _ in RecallModel(CFG, "softmax").named_parameters()}
    expected = {"ca.v.weight", "ca.out.weight", "ca.out.bias", "ff2.0.weight", "ff2.0.bias", "ff2.2.weight",
                "ff2.2.bias", "norms.4.weight", "norms.4.bias", "norms.5.weight", "norms.5.bias", "head.weight",
                "head.bias"}
    assert {n for n in names if n.startswith(e5.REFIT_PREFIXES)} == expected
    frozen = names - expected
    for prefix in ("key_in.", "val_in.", "sa.q.", "sa.k.", "sa.v.", "sa.out.", "ff1.", "norms.0.", "norms.1.",
                   "norms.2.", "norms.3.", "ca.q.", "ca.k."):
        assert any(n.startswith(prefix) for n in frozen), prefix
    assert "query_type" in frozen


def test_refit_freezes_scores_and_trains_the_rest():
    data = batch()
    for t, i in PAIRS:
        model = trained(t)
        original = copy.deepcopy(model.state_dict())
        plain = e5.view(model, i, CFG)
        before = copy.deepcopy(plain.state_dict())
        fitted = copy.deepcopy(plain)
        info = e5.refit(fitted, CFG, SEED, NOISE)
        assert info["nonfinite_steps"] == 0
        after = fitted.state_dict()
        for k in before:
            same = torch.equal(before[k], after[k])
            assert same != k.startswith(e5.REFIT_PREFIXES), (t, i, k, same)
        assert all(torch.equal(original[k], v) for k, v in model.state_dict().items()), "original changed"
        assert torch.equal(cross_scores(plain, data), cross_scores(fitted, data)), (t, i)
        assert (e5.retrieval(fitted, t, CFG, "test", SEED, NOISE, N, 1)
                == e5.retrieval(model, t, CFG, "test", SEED, NOISE, N, 1)), (t, i)
        assert not torch.equal(output(plain, data), output(fitted, data)), (t, i)


def test_check_frozen_fires():
    model = trained("softmax")
    snapshot = e5.frozen_snapshot(model)
    e5.check_frozen(model, snapshot)
    with torch.no_grad():
        model.ca.k.weight.add_(1e-3)
    try:
        e5.check_frozen(model, snapshot)
    except AssertionError:
        return
    raise AssertionError("check_frozen missed a change")


def test_test_data_is_paired_and_streams_are_disjoint():
    a, b = batch(), batch()
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    results = [e5.retrieval(trained(t), t, CFG, "test", SEED, NOISE, N, 2)["matched_filter_acc"] for t in e5.ARMS]
    assert len(set(results)) == 1
    n = CFG["train_n"]
    keys = [e5.make_batch(e5.data_gen(kind, SEED, NOISE, n, CFG), 4, n, NOISE, CFG)[0]
            for kind in ("train", "refit", "val", "test")]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.equal(keys[i], keys[j])


def test_seeds_are_disjoint():
    e1_1 = {0, 100, 101, 102, 103, 104}
    final, tune = set(e5.FULL["final_seeds"]), set(e5.FULL["tune_seeds"])
    assert len(final) == 5 and not (final & e1_1) and not (tune & e1_1) and not (final & tune)
    assert e5.FULL["steps"] == 2500 and e5.FULL["train_n"] == 8 and e5.FULL["factors"] == [1, 8, 32]
    assert e5.FULL["lrs"] == [3e-3, 6e-3, 1e-2] and e5.FULL["noises"] == [0.6, 0.9]


def test_select_lrs():
    cfg = dict(lrs=[3e-3, 6e-3, 1e-2], noises=[0.6], tune_seeds=[300])
    rows = {3e-3: (0.9, 0.5), 6e-3: (0.95, 0.4), 1e-2: (0.95, 0.3)}  # tie on accuracy: lower loss
    records = {(a, 300, lr, 0.6): dict(val_acc=v[0], val_loss=v[1]) for a in e5.ARMS for lr, v in rows.items()}
    assert e5.select_lrs(records, cfg)[("rem", 0.6)][0] == 1e-2
    rows = {3e-3: (0.9, 0.5), 6e-3: (0.9, 0.5), 1e-2: (0.9, 0.5)}  # full tie: smaller lr
    records = {(a, 300, lr, 0.6): dict(val_acc=v[0], val_loss=v[1]) for a in e5.ARMS for lr, v in rows.items()}
    assert e5.select_lrs(records, cfg)[("softmax", 0.6)][0] == 3e-3


def test_share_estimators():
    s = an.share_summary({1: 0.06, 2: 0.06, 3: 0.06}, {1: 0.08, 2: 0.08, 3: 0.08})
    assert abs(s["mean"] - 0.75) < 1e-9 and s["method"].startswith("per-seed")
    s = an.share_summary({1: 0.04, 2: 0.05, 3: 0.03, 4: 0.04}, {1: 0.08, 2: 0.10, 3: 0.01, 4: 0.09})
    assert s["method"].startswith("ratio of means")
    assert abs(s["mean"] - 0.16 / 0.28) < 1e-9 and s["lo"] <= s["mean"] <= s["hi"]
    s = an.share_summary({1: 0.04, 2: 0.04, 3: 0.04}, {1: 0.05, 2: -0.05, 3: 0.01})
    assert s["lo"] == -math.inf and s["hi"] == math.inf and "unbounded" in s["method"]


def test_reading_rules():
    def share(lo, hi):
        return dict(lo=lo, hi=hi)

    def ret(mean, lo):
        return dict(mean=mean, lo=lo)

    assert an.read_noise(share(0.6, 0.9), ret(0.005, -0.01)) == "INFERENCE DOMINATES"
    assert an.read_noise(share(0.6, 0.9), ret(0.02, 0.01)) == "MIXED OR UNRESOLVED"
    assert an.read_noise(share(0.1, 0.4), ret(0.02, 0.01)) == "TRAINING DOMINATES"
    assert an.read_noise(share(0.1, 0.4), ret(0.02, -0.01)) == "MIXED OR UNRESOLVED"
    assert an.read_noise(share(0.3, 0.7), ret(0.0, -0.01)) == "MIXED OR UNRESOLVED"
    assert an.read_noise(share(math.nan, math.nan), ret(math.nan, math.nan)) == "MIXED OR UNRESOLVED"
    assert an.premise_failed([dict(lo=-0.1, hi=0.1), dict(lo=-0.1, hi=0.1)])
    assert not an.premise_failed([dict(lo=-0.1, hi=0.1), dict(lo=0.01, hi=0.1)])


def test_run_cell_to_analysis():
    """The record schema written by run.py is what analyze.py reads."""
    cfg = dict(CFG, eval_batch=2, val_batches=1, test_batches=1, steps=2, refit_steps=2)

    def task(arm, seed, stage):
        return dict(cfg=cfg, cfg_hash="x", arm=arm, seed=seed, lr=3e-3, noise=NOISE, stage=stage,
                    lr_val_acc={"0.003": 0.5})

    tune = e5.run_cell(task("softmax", 300, "tune"))
    assert "error" not in tune, tune.get("error")
    assert "evals" not in tune and "retrieval" not in tune, "the LR check must not touch test data"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.jsonl"
        for seed in (200, 201):
            for arm in e5.ARMS:
                rec = e5.run_cell(task(arm, seed, "final"))
                assert "error" not in rec, rec.get("error")
                assert set(rec["evals"]) == set(e5.readouts_for(arm))
                e5.e11.append_line(path, rec)
        e5.e11.append_line(path, tune)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            an.main(str(path))
    assert "Reading" in out.getvalue() and "pairing" in out.getvalue()


if __name__ == "__main__":
    failed = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

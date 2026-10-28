"""LAB 06: transparent, single-head NumPy attention and calculation checkers.

Rows of Q are queries; rows of K/V are matching key/value positions.
Every public attention call returns (scores, weights, context), with 2-D
outputs even when Q is supplied as one vector. No pretrained semantics.
Run: python attention_numpy.py
"""
import numpy as np


def _matrix(value, name, allow_vector=False):
    array = np.asarray(value, dtype=np.float64)
    if allow_vector and array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a non-empty 2-D matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def stable_softmax(scores, axis=-1, temperature=1.0, mask=None):
    """Stable softmax; True mask entries are allowed. Reject all-masked rows.

    Subtract max before temperature division to avoid unnecessary overflow.
    The default axis normalizes over key positions.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 0 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be non-empty, finite and non-scalar")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    allowed = np.ones_like(scores, dtype=bool) if mask is None else np.broadcast_to(
        np.asarray(mask, dtype=bool), scores.shape)
    if not np.all(allowed.any(axis=axis)):
        raise ValueError("Each query must have at least one unmasked key")
    masked = np.where(allowed, scores, -np.inf)
    shifted = (masked - masked.max(axis=axis, keepdims=True)) / temperature
    exp_scores = np.exp(shifted)
    weights = exp_scores / exp_scores.sum(axis=axis, keepdims=True)
    if not np.isfinite(weights).all():
        raise FloatingPointError("Non-finite softmax output")
    return weights


def attention_scores(Q, K, scaled=False):
    """Return Q @ K.T, optionally divided by sqrt(key dimension)."""
    Q, K = _matrix(Q, "Q", True), _matrix(K, "K")
    if Q.shape[1] != K.shape[1]:
        raise ValueError("Q and K must share their feature dimension")
    scores = Q @ K.T
    if scaled:
        scores = scores / np.sqrt(K.shape[1])
    if not np.isfinite(scores).all():
        raise FloatingPointError("QK.T overflowed; reduce input magnitude")
    return scores


def attention_weights(scores, temperature=1.0, mask=None):
    """Normalize scores over keys."""
    return stable_softmax(scores, axis=-1, temperature=temperature, mask=mask)


def context_vector(weights, V):
    """Weighted combination A @ V for row-normalized attention A."""
    weights, V = _matrix(weights, "weights", True), _matrix(V, "V")
    if weights.shape[1] != V.shape[0]:
        raise ValueError("Number of weights must match the number of values")
    if np.any(weights < 0) or not np.allclose(weights.sum(-1), 1.0):
        raise ValueError("Attention weights must be non-negative and sum to one")
    return weights @ V


def dot_product_attention(Q, K, V, *, temperature=1.0, mask=None):
    """Unscaled dot-product attention, optionally masked or tempered."""
    scores = attention_scores(Q, K, scaled=False)
    weights = attention_weights(scores, temperature, mask)
    return scores, weights, context_vector(weights, V)


def scaled_dot_product_attention(Q, K, V):
    """Compute S=QK.T/sqrt(d_k), A=softmax(S), C=AV from scratch."""
    scores = attention_scores(Q, K, scaled=True)
    weights = attention_weights(scores)
    context = context_vector(weights, V)
    assert weights.shape == scores.shape
    assert context.shape == (scores.shape[0], np.asarray(V).shape[1])
    assert np.allclose(weights.sum(axis=-1), 1.0)
    return scores, weights, context


def self_attention(X, W_Q, W_K, W_V):
    """Project the same X to Q,K,V; return Q,K,V,S,A,C.

    X:(T,d_in); W_Q/W_K:(d_in,d_k); W_V:(d_in,d_v).
    S,A:(T,T); C:(T,d_v).
    """
    X = _matrix(X, "X")
    W_Q, W_K, W_V = [_matrix(w, n) for w, n in
                      [(W_Q, "W_Q"), (W_K, "W_K"), (W_V, "W_V")]]
    if any(w.shape[0] != X.shape[1] for w in [W_Q, W_K, W_V]):
        raise ValueError("Projection input dimensions must match X")
    if W_Q.shape[1] != W_K.shape[1]:
        raise ValueError("Q and K projection dimensions must match")
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    return Q, K, V, *scaled_dot_product_attention(Q, K, V)


def attention_matrix(X, W_Q, W_K, W_V):
    """Return the row-stochastic (T,T) self-attention matrix."""
    return self_attention(X, W_Q, W_K, W_V)[4]


def entropy(weights, axis=-1):
    """Shannon entropy in nats, with 0*log(0) treated as zero."""
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or not np.allclose(weights.sum(axis), 1):
        raise ValueError("Expected normalized non-negative probabilities")
    return -(weights * np.log(np.clip(weights, np.finfo(float).tiny, 1))).sum(axis)


def check_calculation(student_value, reference, atol=5e-4):
    """Check a student's recorded answer without printing the reference answer.

    Invoke after doing calculations.md by hand; a missing answer is not passed.
    """
    if student_value is None:
        return "Chưa kiểm tra: hãy ghi lời giải bằng tay trước."
    student_value = np.asarray(student_value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if student_value.shape != reference.shape:
        return f"Sai shape: cần {reference.shape}, nhận {student_value.shape}."
    return "Khớp trong sai số cho phép." if np.allclose(
        student_value, reference, atol=atol, rtol=0) else "Chưa khớp; tự kiểm tra từng bước."


def run_unit_tests():
    """Dataset-free mathematical tests. Return names of checks that passed."""
    passed = []
    def check(name, condition):
        assert condition, name
        passed.append(name)
    s = np.array([1000., 1001., 1002.])
    a = stable_softmax(s)
    check("large-score softmax finite", np.isfinite(a).all())
    check("softmax sums to one", np.isclose(a.sum(), 1))
    check("softmax shift invariance", np.allclose(a, stable_softmax(s - 1000)))
    check("equal scores uniform", np.allclose(stable_softmax(np.ones(4)), .25))
    check("temperature entropy ordering", entropy(stable_softmax(s, temperature=.5)) <
          entropy(a) < entropy(stable_softmax(s, temperature=2)))
    rng = np.random.default_rng(7)
    Q, K, V = rng.normal(size=(2, 4)), rng.normal(size=(3, 4)), rng.normal(size=(3, 5))
    S, A, C = scaled_dot_product_attention(Q, K, V)
    reference_scores = np.array([[np.dot(q, k)/2 for k in K] for q in Q])
    ref_weights = np.exp(reference_scores - reference_scores.max(-1, keepdims=True))
    ref_weights /= ref_weights.sum(-1, keepdims=True)
    ref_context = np.array([sum(w * v for w, v in zip(row, V)) for row in ref_weights])
    check("attention scores match loop", np.allclose(S, reference_scores))
    check("attention weights match reference", np.allclose(A, ref_weights))
    check("context matches weighted sum", np.allclose(C, ref_context))
    check("attention shapes", S.shape == A.shape == (2, 3) and C.shape == (2, 5))
    check("vector query promoted", scaled_dot_product_attention(Q[0], K, V)[2].shape == (1, 5))
    check("unscaled scores", np.allclose(dot_product_attention(Q, K, V)[0], Q @ K.T))
    check("PAD-key mask", np.allclose(stable_softmax(S, mask=[1, 1, 0])[:, 2], 0))
    X = rng.normal(size=(4, 3))
    WQ, WK, WV = rng.normal(size=(3, 2)), rng.normal(size=(3, 2)), rng.normal(size=(3, 5))
    q, k, v, scores, weights, contexts = self_attention(X, WQ, WK, WV)
    check("self-attention projections", np.allclose(q, X @ WQ) and np.allclose(k, X @ WK)
          and np.allclose(v, X @ WV))
    check("self-attention row sums", weights.shape == (4, 4) and np.allclose(weights.sum(-1), 1))
    check("self-attention context", np.allclose(contexts, weights @ v))
    permutation = np.array([2, 0, 3, 1])
    p = self_attention(X[permutation], WQ, WK, WV)
    check("permutation equivariance", np.allclose(p[5], contexts[permutation]))
    check("attention_matrix helper", np.allclose(attention_matrix(X, WQ, WK, WV), weights))
    for name, call in [
        ("reject incompatible Q/K", lambda: attention_scores(Q, K[:, :2])),
        ("reject incompatible K/V", lambda: scaled_dot_product_attention(Q, K, V[:2])),
        ("reject all-masked rows", lambda: stable_softmax(S, mask=np.zeros_like(S, dtype=bool))),
        ("reject invalid temperature", lambda: stable_softmax(S, temperature=0)),
        ("reject non-finite input", lambda: stable_softmax([np.nan, 1])),
    ]:
        try:
            call()
        except ValueError:
            passed.append(name)
        else:
            raise AssertionError(name)
    return passed


if __name__ == "__main__":
    checks = run_unit_tests()
    print(f"PASS: {len(checks)} NumPy checks")
    for name in checks:
        print(f"  - {name}")

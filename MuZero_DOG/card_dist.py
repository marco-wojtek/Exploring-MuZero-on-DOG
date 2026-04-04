from math import comb
import numpy as np
import jax
import jax.numpy as jnp
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from DOG.dog import *
jax.config.update("jax_enable_x64", True)
JOKER_IDX      = 0
SWAP_IDX       = 1
NEG4_IDX       = 4   # 4-card: uniquely enables the -4 backward move into the goal area
SEVEN_IDX      = 7
ONE_ELEVEN_IDX = 11
THIRTEEN_IDX   = 13
NORMAL_IDXS    = jnp.array([2, 3, 5, 6, 8, 9, 10, 12])  # pure forward-only blue cards

def presence_prob_array(env:DOG):
    """
    deck_counts: array of k integers, Anzahl Karten pro Typ im Deck
    N: Anzahl gezogener Karten

    Returns:
        np.ndarray of shape (8192,) = (2^13,), dtype float64.
        Index is a 13-bit bitmask: bit t is 1 iff card type t is present.
        result[mask] = probability that exactly the types indicated by mask
        are present in the N drawn cards.

    Uses Möbius inversion on the Boolean subset lattice via a
    Sum-Over-Subsets (SOS) DP, reducing complexity from O(3^k) to O(k·2^k).

    Let Q[A] = C(cards_in_A, N) / C(T, N) = P(all N draws come from types in A).
    Then P[B] = Σ_{A ⊆ B} (−1)^(|B|−|A|) · Q[A], computed with the SOS DP.
    """
    is_card_dist = (env.phase == 0) and jnp.all(env.swap_choices == -1)  # Nur im Phase 1, wenn noch keine Swaps stattgefunden haben, ist die Kartenverteilung relevant

    if not is_card_dist:
        # Wenn kein chance event, dann dummy array mit element an 0 mit 100%
        result = np.zeros(1 << 13, dtype=np.float64)
        result[0] = 1.0
        return result
    
    deck_counts = env.deck  # Anzahl Karten pro Typ im Deck
    N = env.hand_size  # Anzahl zu ziehender Karten (Handgröße)
    T = sum(deck_counts)
    k = len(deck_counts)
    n_masks = 1 << k  # 2^k

    # Precompute C(r, N) / C(T, N) for r = 0 … T
    binom_T_N = comb(T, N)
    binom_vals = jnp.array(
        [comb(r, N) / binom_T_N if r >= N else 0.0 for r in range(T + 1)],
        dtype=jnp.float64,
    )

    deck_arr = jnp.array(deck_counts, dtype=jnp.int32)
    m_idx = jnp.arange(n_masks, dtype=jnp.int32)  # shape (2^k,)

    # bits[m, t] = (m >> t) & 1  →  shape (2^k, k)
    bits = (m_idx[:, None] >> jnp.arange(k, dtype=jnp.int32)[None, :]) & 1

    # removed[m] = total cards belonging to the types in subset m
    removed = (bits @ deck_arr).astype(jnp.int32)

    # Q[m] = C(removed[m], N) / C(T, N)  =  P(all N draws come from types in m)
    Q = binom_vals[removed]

    # sign[m] = (-1)^|m|  where |m| is the popcount of m
    popcount = bits.sum(axis=1)
    sign = jnp.where(popcount % 2 == 0, 1.0, -1.0).astype(jnp.float64)

    # SOS DP (Möbius zeta transform):  F[B] = Σ_{A ⊆ B}  sign[A] · Q[A]
    F = sign * Q
    for i in range(k):
        mask_bit = jnp.int32(1 << i)
        m_prev = m_idx & ~mask_bit        # index of m with bit i cleared
        has_bit = (m_idx & mask_bit) != 0
        F = jnp.where(has_bit, F + F[m_prev], F)

    MAX_CARD_TYPES = 14
    # Möbius inversion:  P[B] = sign[B] · F[B]
    P_local = np.asarray(sign * F)  # shape (2^k,)

    # Embed into fixed-size output of length 2^13 = 8192.
    # Bits k..12 are unused (those card types don't exist), so their
    # probability is 0 — correct for any mask that has those bits set.
    result = np.zeros(1 << MAX_CARD_TYPES, dtype=np.float64)
    result[:n_masks] = P_local
    return result

# Map from 14-bit deck-prior → 128-d category distribution (run in numpy, outside JAX)
def card_dist_128(env: DOG) -> np.ndarray:
    """
    Returns float32[128]: P(current player's dealt hand falls in each 7-bit category).
    Bit layout: 0=joker 1=swap 2=13 3=7 4=1/11 5=4-card(-4 move) 6=any-normal-fwd
    Index 0 = no deal (dummy). Must only be called outside JAX JIT.
    """
    full_16384 = presence_prob_array(env)
    result = np.zeros(128, dtype=np.float64)
    for mask14 in range(1 << 14):
        if full_16384[mask14] == 0.0:
            continue
        cat = _mask14_to_cat7(mask14)
        result[cat] += full_16384[mask14]
    return result.astype(np.float32)

# Helper: collapse 14-bit presence mask → 7-bit strategic category mask
# Bit 0: joker(idx 0)   Bit 1: swap(idx 1)    Bit 2: 13(idx 13)   Bit 3: 7(idx 7)
# Bit 4: 1/11(idx 11)   Bit 5: 4-card(idx 4)  Bit 6: normal-fwd(2,3,5,6,8,9,10,12)
def _mask14_to_cat7(mask14: int) -> int:
    NORMAL_IDXS_PY = [2, 3, 5, 6, 8, 9, 10, 12]
    bits = [
        (mask14 >> JOKER_IDX)      & 1,
        (mask14 >> SWAP_IDX)       & 1,
        (mask14 >> THIRTEEN_IDX)   & 1,
        (mask14 >> SEVEN_IDX)      & 1,
        (mask14 >> ONE_ELEVEN_IDX) & 1,
        (mask14 >> NEG4_IDX)       & 1,
        any((mask14 >> idx) & 1 for idx in NORMAL_IDXS_PY),
    ]
    return int(sum(b << i for i, b in enumerate(bits)))



def compute_card_outcome_jax(env_after: DOG, player):
    """Returns int32 in 0..127: 7-bit category mask of cards player holds.
    Bit 5 is set if the 4-card is present (enables unique -4 backward move).
    Returns 0 (dummy) only when no deal happened (caller's responsibility).
    Pure JAX — safe to call inside jax.lax.while_loop / jax.jit.
    """
    h = env_after.hands[player]
    bits = jnp.stack([
        h[JOKER_IDX]              > 0,
        h[SWAP_IDX]               > 0,
        h[THIRTEEN_IDX]           > 0,
        h[SEVEN_IDX]              > 0,
        h[ONE_ELEVEN_IDX]         > 0,
        h[NEG4_IDX]               > 0,   # 4-card: only card enabling -4 move
        jnp.any(h[NORMAL_IDXS]   > 0),  # any pure forward blue card
    ]).astype(jnp.int32)
    powers = jnp.array([1, 2, 4, 8, 16, 32, 64], dtype=jnp.int32)
    return jnp.sum(bits * powers)
import numpy as np
import os, glob

# Auto-detect latest replay_diag file, or set PATH explicitly
files = sorted(glob.glob('MuZero_DOG/models/replay_diag_*.npz'))
PATH = files[-1] if files else 'MuZero_DOG/models/replay_diag_....npz'
print(f"Loading: {PATH}\n")

d = np.load(PATH)
print(f"Keys: {list(d.keys())}\n")

def sep(title=""):
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")

rewards    = d['rewards'].astype(np.int8)          # (episodes, max_ep_len)  0=loss 1=neutral 2=win
masks      = d['masks'].astype(bool)               # (episodes, max_ep_len)
card_dists = d['card_distributions'].astype(np.float32)  # (episodes, max_ep_len, 128)
ep_lengths = d['episode_lengths'].astype(np.int32) # (episodes,)
players    = d['players'].astype(np.int8)          # (episodes, max_ep_len)
has_pol    = 'policies' in d

n_ep = len(ep_lengths)
cap  = int(rewards.shape[1])

# ── flat views of REAL steps only ──────────────────────────────
rew_real  = rewards[masks]    # (N_real,)
plr_real  = players[masks]    # (N_real,)
N_real    = len(rew_real)

sep(f"OVERVIEW  —  {n_ep} episodes, {N_real:,} real steps (mask=True)")
print(f"  max_ep_len          : {cap}")
print(f"  valid fraction      : {masks.mean():.4f}  ({N_real:,} / {masks.size:,})")


# ── Episode lengths ─────────────────────────────────────────────
sep("EPISODE LENGTHS")
print(f"  min={ep_lengths.min()}  max={ep_lengths.max()}  "
      f"mean={ep_lengths.mean():.1f}  median={int(np.median(ep_lengths))}")
print(f"  pct at cap ({cap})   : {100*(ep_lengths >= cap).mean():.1f}%")


# ── Game outcomes (final step of each episode) ──────────────────
sep("GAME OUTCOMES  (final reward of each episode)")
final_rews = rewards[np.arange(n_ep), ep_lengths - 1]
n_loss    = int(np.sum(final_rews == 0))
n_neutral = int(np.sum(final_rews == 1))
n_win     = int(np.sum(final_rews == 2))
print(f"  loss    (0) : {n_loss:>6}  ({100*n_loss/n_ep:.1f}%)")
print(f"  neutral (1) : {n_neutral:>6}  ({100*n_neutral/n_ep:.1f}%)   hit max_steps / timed out")
print(f"  win     (2) : {n_win:>6}  ({100*n_win/n_ep:.1f}%)")


# ── Reward class-2 position audit ──────────────────────────────
sep("REWARD CLASS-2 POSITION AUDIT")
# For each episode, find every position where reward==2 (within the real episode, not padding)
mid_game_class2 = 0   # class-2 at any step BEFORE the final step
final_step_class2 = 0  # class-2 exactly at the final step
multi_class2 = 0       # episodes with >1 class-2 entry
zero_class2  = 0       # episodes with no class-2 entry at all
class2_rel_positions = []  # relative position (step / ep_length) of each class-2 hit

for i in range(n_ep):
    L = int(ep_lengths[i])
    ep_rew = rewards[i, :L]           # raw reward sequence (includes no_step = mask=0)
    final_idx = L - 1
    positions = np.where(ep_rew == 2)[0]  # all indices with class-2 in this episode
    n_hits = len(positions)

    if n_hits == 0:
        zero_class2 += 1
    if n_hits > 1:
        multi_class2 += 1

    for p in positions:
        class2_rel_positions.append(p / final_idx if final_idx > 0 else 1.0)
        if p < final_idx:
            mid_game_class2 += 1
        else:
            final_step_class2 += 1

total_class2 = mid_game_class2 + final_step_class2
print(f"  Total class-2 entries in buffer : {total_class2:,}")
print(f"  At the final step (ep_len-1)    : {final_step_class2:,}  ({100*final_step_class2/max(total_class2,1):.1f}%)")
print(f"  BEFORE the final step (MID-GAME): {mid_game_class2:,}  ({100*mid_game_class2/max(total_class2,1):.1f}%)   should be 0")
print(f"  Episodes with ZERO class-2      : {zero_class2:,}  ({100*zero_class2/n_ep:.1f}%)")
print(f"  Episodes with >1 class-2        : {multi_class2:,}  ({100*multi_class2/n_ep:.1f}%)")
if class2_rel_positions:
    rel = np.array(class2_rel_positions)
    print(f"  Relative position (0=start,1=end): mean={rel.mean():.4f}  min={rel.min():.4f}  max={rel.max():.4f}")
    # Histogram of relative positions
    bins = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
    for lo, hi in zip(bins, bins[1:]):
        cnt = int(np.sum((rel >= lo) & (rel < hi if hi < 1.0 else rel <= hi)))
        print(f"    [{lo:.2f}, {hi:.2f}{')'if hi<1.0 else']'} : {cnt:>6,}")
has_players = 'players' in d
if mid_game_class2 > 0:
    print(f"\n  !! MID-GAME CLASS-2 DETECTED — showing first 20 examples:")
    # Also tally: how many mid-game class-2 have mask=True vs mask=False
    mid_masked   = 0  # real MCTS step
    mid_unmasked = 0  # no_step slot
    shown = 0
    for i in range(n_ep):
        L = int(ep_lengths[i])
        ep_rew  = rewards[i, :L]
        ep_mask = masks[i, :L]
        ep_plr  = d['players'][i, :L] if has_players else None
        positions = np.where(ep_rew == 2)[0]
        for p in positions:
            if p < L - 1:
                if ep_mask[p]:
                    mid_masked += 1
                else:
                    mid_unmasked += 1
                if shown < 20:
                    lo, hi = max(0, p-3), min(L, p+4)
                    rew_ctx  = rewards[i, lo:hi].tolist()
                    mask_ctx = masks[i, lo:hi].tolist()
                    plr_ctx  = d['players'][i, lo:hi].tolist() if has_players else '?'
                    print(f"    ep={i:4d} len={L:4d} step={p:4d}/{L-1:4d} "
                          f"mask={bool(ep_mask[p])}  "
                          f"rewards[{lo}:{hi}]={rew_ctx}  "
                          f"masks[{lo}:{hi}]={mask_ctx}  "
                          f"players[{lo}:{hi}]={plr_ctx}")
                    shown += 1
    print(f"\n  Mid-game class-2 breakdown:")
    print(f"    mask=True  (real MCTS step) : {mid_masked:,}  - DEFINITE BUG")
    print(f"    mask=False (no_step slot)   : {mid_unmasked:,}  - no_step returned reward>0 (also a bug)")
else:
    print(f"\n  ✓  All class-2 entries are at the final step — no mid-game wins.")


# ── Reward classes — mask-filtered (mid-game steps) ────────────
sep("REWARD CLASSES  (real steps only, mask=True)")
for cls, label in [(0,'loss'), (1,'neutral'), (2,'win')]:
    cnt = int(np.sum(rew_real == cls))
    print(f"  class {cls} ({label:>7}) : {cnt:>8,}  ({100*cnt/N_real:.2f}%)")
print(f"  NOTE: mid-game should be almost all class-1 (neutral).")
print(f"        Non-trivial class-0/2 mid-game = potential reward assignment bug.")


# ── Player distribution — mask-filtered ────────────────────────
sep("PLAYER DISTRIBUTION  (real steps only)")
total_plr = 0
for i in range(4):
    cnt = int(np.sum(plr_real == i))
    total_plr += cnt
    print(f"  Player {i}: {cnt:>8,}  ({100*cnt/N_real:.1f}%)")
print(f"  (balanced = ~25% each; large imbalance = encoding bug)")


# ── Card distributions — mask-filtered ─────────────────────────
sep("CARD DEAL STEPS  (real steps only, mask=True)")
# A real deal step has a non-degenerate distribution (first bin < 0.99)
cdist_real = card_dists[masks]                     # (N_real, 128)
deal_real  = cdist_real[:, 0] < 0.99
n_deals    = int(deal_real.sum())
print(f"  Deal steps  : {n_deals:,}  ({100*n_deals/N_real:.2f}% of real steps)")
print(f"  Non-deal    : {N_real - n_deals:,}")
if n_deals > 0:
    deal_dists = cdist_real[deal_real]
    avg_dist   = deal_dists.mean(axis=0)
    top5       = np.argsort(-avg_dist)[:5]
    print(f"  Top-5 card bins (by avg prob) : {top5.tolist()}")
    print(f"  Their avg probabilities       : {np.round(avg_dist[top5], 4).tolist()}")
    deals_per_game = n_deals / n_ep
    print(f"  Avg deal steps / episode      : {deals_per_game:.1f}")


# ── Policy / MCTS-visit stats — mask-filtered ──────────────────
sep("MCTS POLICY STATS  (real steps only)")
if has_pol:
    pols = d['policies'].astype(np.float32)   # (n_ep, max_len, 454)
    pol_real = pols[masks]                     # (N_real, 454)

    # Visited-action count = actions with any visit weight > 0
    visited = np.sum(pol_real > 0, axis=-1)    # (N_real,)
    print(f"  MCTS-visited actions/step : mean={visited.mean():.1f}  "
          f"min={visited.min()}  max={visited.max()}  median={int(np.median(visited))}")
    print(f"  NOTE: this is NOT the number of legal actions,")
    print(f"        but how many distinct actions MCTS explored (capped by sim budget).")

    buckets = [(1,1,'=1'), (2,5,'2-5'), (6,10,'6-10'), (11,20,'11-20'),
               (21,50,'21-50'), (51,100,'51-100'), (101,454,'101+')]
    for lo, hi, lbl in buckets:
        cnt = int(np.sum((visited >= lo) & (visited <= hi)))
        if cnt > 0:
            print(f"    {lbl:>8} : {cnt:>8,}  ({100*cnt/N_real:.1f}%)")

    # Policy entropy (low = concentrated on one action)
    pol_c   = np.clip(pol_real, 1e-9, 1.0)
    entropy = -np.sum(pol_real * np.log(pol_c), axis=-1)
    print(f"  Policy entropy (nats)     : mean={entropy.mean():.3f}  "
          f"min={entropy.min():.3f}  max={entropy.max():.3f}")
    print(f"  Reference: uniform-2={np.log(2):.2f}  uniform-5={np.log(5):.2f}  "
          f"uniform-10={np.log(10):.2f}  uniform-454={np.log(454):.2f}")

    # Max probability per step (how dominant is the top action)
    max_prob = pol_real.max(axis=-1)
    print(f"  Top-action prob           : mean={max_prob.mean():.3f}  "
          f"min={max_prob.min():.3f}  max={max_prob.max():.3f}")
    print(f"  (close to 1.0 = policy collapsed; ~0.5 with 2 legal moves = healthy)")

    # Most visited actions overall
    mean_pol = pol_real.mean(axis=0)
    top5_act = np.argsort(-mean_pol)[:5]
    print(f"  Top-5 actions by avg weight: {top5_act.tolist()}")
    print(f"  Their avg weights          : {np.round(mean_pol[top5_act], 4).tolist()}")
else:
    print("  No 'policies' key in file.")


# ── Per-episode summary (first 30) ─────────────────────────────
sep("PER-EPISODE DETAIL  (first 30)")
hdr = f"{'Ep':>4}  {'Len':>5}  {'FinalR':>7}  {'RealSteps':>9}  {'Deals':>6}"
if has_pol:
    hdr += f"  {'VisitedActs':>11}  {'Entropy':>8}  {'TopActProb':>10}"
print(hdr)
for i in range(min(30, n_ep)):
    L  = int(ep_lengths[i])
    fr = int(rewards[i, L - 1])
    ep_mask = masks[i, :L]
    rs = int(ep_mask.sum())
    dl = int(np.sum(card_dists[i, :L][ep_mask, 0] < 0.99)) if rs > 0 else 0
    row = f"{i:>4}  {L:>5}  {fr:>7}  {rs:>9}  {dl:>6}"
    if has_pol and rs > 0:
        ep_pol = pols[i, :L][ep_mask]
        vis    = np.sum(ep_pol > 0, axis=-1).mean()
        ep_c   = np.clip(ep_pol, 1e-9, 1.0)
        ent    = (-np.sum(ep_pol * np.log(ep_c), axis=-1)).mean()
        top_p  = ep_pol.max(axis=-1).mean()
        row   += f"  {vis:>11.1f}  {ent:>8.3f}  {top_p:>10.3f}"
    print(row)

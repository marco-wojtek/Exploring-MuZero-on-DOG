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

rewards   = d['rewards'].astype(np.int8)           # (sessions, max_sess_len)  0=loss 1=neutral 2=win
masks     = d['masks'].astype(bool)                # (sessions, max_sess_len)
tgts      = d['target_values'].astype(np.float32)  # (sessions, max_sess_len)
disc      = d['discount_targets'].astype(np.int8)  # (sessions, max_sess_len)
ep_lens   = d['episode_lengths'].astype(np.int32)  # (sessions,)
has_pol   = 'policies' in d

n_sess  = len(ep_lens)
max_len = int(rewards.shape[1])

# flat views of REAL steps only
rew_real  = rewards[masks]
tgt_real  = tgts[masks]
N_real    = len(rew_real)

sep(f"OVERVIEW  —  {n_sess} sessions, {N_real:,} real steps (mask=True)")
print(f"  max_session_len     : {max_len}")
print(f"  valid fraction      : {masks.mean():.4f}  ({N_real:,} / {masks.size:,})")


# ── Session lengths ─────────────────────────────────────────────
sep("SESSION LENGTHS")
print(f"  min={ep_lens.min()}  max={ep_lens.max()}  "
      f"mean={ep_lens.mean():.1f}  median={int(np.median(ep_lens))}")

# Fill rate: how much of max_session_len is used on average
fill_pct = ep_lens / max_len * 100
print(f"  Fill rate (len/max) : mean={fill_pct.mean():.1f}%  "
      f"min={fill_pct.min():.1f}%  max={fill_pct.max():.1f}%")

# Length histogram
bins = [1, 4, 8, 12, 16, 20, 24, 28]
print("  Length distribution:")
for lo, hi in zip(bins, bins[1:]):
    cnt = int(np.sum((ep_lens >= lo) & (ep_lens < hi)))
    bar = '#' * (cnt * 40 // max(n_sess, 1))
    print(f"    [{lo:2d},{hi:2d}) : {cnt:>6,}  {bar}")
cnt_full = int(np.sum(ep_lens == max_len))
print(f"    [={max_len}]  : {cnt_full:>6,}  (at cap)")


# ── Terminal sessions ───────────────────────────────────────────
sep("TERMINAL SESSIONS  (sessions with win or loss reward)")
# A session is terminal if any of its valid steps has reward class 0 or 2
has_win  = np.any((rewards == 2) & masks, axis=1)
has_loss = np.any((rewards == 0) & masks, axis=1)
n_win    = int(has_win.sum())
n_loss   = int(has_loss.sum())
n_term   = int((has_win | has_loss).sum())
print(f"  Win  sessions : {n_win:>6,}  ({100*n_win/n_sess:.1f}%)")
print(f"  Loss sessions : {n_loss:>6,}  ({100*n_loss/n_sess:.1f}%)")
print(f"  Terminal total: {n_term:>6,}  ({100*n_term/n_sess:.1f}%)  ← ~10% expected (1 of ~10 sessions/game)")
print(f"  Non-terminal  : {n_sess-n_term:>6,}  ({100*(n_sess-n_term)/n_sess:.1f}%)")

# Where in the session does the win/loss reward appear?
term_positions = []
for i in range(n_sess):
    L = int(ep_lens[i])
    pos = np.where((rewards[i, :L] == 0) | (rewards[i, :L] == 2))[0]
    for p in pos:
        term_positions.append(p / max(L - 1, 1))
if term_positions:
    rel = np.array(term_positions)
    print(f"\n  Relative position of win/loss reward (0=start, 1=end):")
    print(f"    mean={rel.mean():.3f}  min={rel.min():.3f}  max={rel.max():.3f}")
    print(f"    Values near 1.0 = reward at session end (correct)")
    bins_r = [0.0, 0.5, 0.8, 0.95, 1.0]
    for lo, hi in zip(bins_r, bins_r[1:]):
        cnt = int(np.sum((rel >= lo) & (rel <= hi)))
        print(f"    [{lo:.2f}, {hi:.2f}] : {cnt:>6,}  {'✓' if hi == 1.0 else '!! should be 0'}")


# ── Reward classes ──────────────────────────────────────────────
sep("REWARD CLASSES  (valid steps only)")
for cls, label in [(0, 'loss'), (1, 'neutral'), (2, 'win')]:
    cnt = int(np.sum(rew_real == cls))
    print(f"  class {cls} ({label:>7}) : {cnt:>8,}  ({100*cnt/max(N_real,1):.2f}%)")


# ── Discount targets ────────────────────────────────────────────
sep("DISCOUNT TARGETS  (checking session-end = terminal)")
# disc shape matches masks shape (both max_sess_len), discount stored for K-1 steps
# but we stored it at all positions — use masks to filter
disc_real = disc[masks]
for cls, label in [(0, 'other_team_turn'), (1, 'terminal/session_end'), (2, 'same_team_turn')]:
    cnt = int(np.sum(disc_real == cls))
    print(f"  class {cls} ({label:>22}) : {cnt:>8,}  ({100*cnt/max(len(disc_real),1):.2f}%)")

# Check: last valid step of each session should be class 1 (discount=0, session terminal)
last_disc = disc[np.arange(n_sess), ep_lens - 1]
n_last_terminal = int(np.sum(last_disc == 1))
print(f"\n  Last step of each session:")
print(f"    discount=1 (terminal) : {n_last_terminal:>6,} / {n_sess}  "
      f"({100*n_last_terminal/n_sess:.1f}%)  ← should be 100%")
if n_last_terminal < n_sess:
    print(f"    !! {n_sess - n_last_terminal} sessions do NOT have terminal discount at last step")
    # Show a few examples
    bad = np.where(last_disc != 1)[0][:5]
    for i in bad:
        L = int(ep_lens[i])
        print(f"      sess={i}  len={L}  disc[-1]={int(last_disc[i])}  "
              f"disc[:L]={disc[i, :L].tolist()}")
else:
    print(f"    ✓ All sessions correctly end with discount=1")


# ── Target values ───────────────────────────────────────────────
sep("TARGET VALUES  (valid steps only)")
print(f"  min={tgt_real.min():.3f}  max={tgt_real.max():.3f}  "
      f"mean={tgt_real.mean():.4f}  std={tgt_real.std():.4f}")
n_zero    = int(np.sum(tgt_real == 0.0))
n_pos     = int(np.sum(tgt_real > 0.0))
n_neg     = int(np.sum(tgt_real < 0.0))
print(f"  positive : {n_pos:>8,}  ({100*n_pos/max(N_real,1):.2f}%)  ← win-side bootstrap")
print(f"  zero     : {n_zero:>8,}  ({100*n_zero/max(N_real,1):.2f}%)  ← neutral / uncertain")
print(f"  negative : {n_neg:>8,}  ({100*n_neg/max(N_real,1):.2f}%)  ← loss-side bootstrap")

# Target value distribution in bins
print("  Histogram:")
tbins = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
for lo, hi in zip(tbins, tbins[1:]):
    cnt = int(np.sum((tgt_real >= lo) & (tgt_real < hi if hi < 1.0 else tgt_real <= hi)))
    bar = '#' * min(cnt * 50 // max(N_real, 1), 40)
    print(f"    [{lo:+.1f}, {hi:+.1f}{')'if hi<1.0 else']'} : {cnt:>8,}  {bar}")

# Per-session: terminal vs non-terminal target value comparison
tgt_term    = tgts[has_win | has_loss][masks[has_win | has_loss]]
tgt_nonterm = tgts[~(has_win | has_loss)][masks[~(has_win | has_loss)]]
if len(tgt_term) > 0 and len(tgt_nonterm) > 0:
    print(f"\n  Terminal sessions   — mean target: {tgt_term.mean():.4f}  std: {tgt_term.std():.4f}")
    print(f"  Non-terminal sess. — mean target: {tgt_nonterm.mean():.4f}  std: {tgt_nonterm.std():.4f}")
    print(f"  (Non-terminal targets closer to 0 = bootstrap working correctly)")


# ── Policy / MCTS-visit stats ───────────────────────────────────
sep("MCTS POLICY STATS  (valid steps only)")
if has_pol:
    pols     = d['policies'].astype(np.float32)   # (n_sess, max_len, 454)
    pol_real = pols[masks]                         # (N_real, 454)

    visited = np.sum(pol_real > 0, axis=-1)
    print(f"  Visited actions/step : mean={visited.mean():.1f}  "
          f"min={visited.min()}  max={visited.max()}  median={int(np.median(visited))}")

    buckets = [(1,1,'=1'), (2,5,'2-5'), (6,10,'6-10'), (11,20,'11-20'), (21,50,'21+')]
    for lo, hi, lbl in buckets:
        cnt = int(np.sum((visited >= lo) & (visited <= hi)))
        print(f"    {lbl:>6} : {cnt:>8,}  ({100*cnt/max(N_real,1):.1f}%)")

    pol_c   = np.clip(pol_real, 1e-9, 1.0)
    entropy = -np.sum(pol_real * np.log(pol_c), axis=-1)
    max_p   = pol_real.max(axis=-1)
    print(f"  Policy entropy (nats): mean={entropy.mean():.3f}  "
          f"min={entropy.min():.3f}  max={entropy.max():.3f}")
    print(f"  Top-action prob      : mean={max_p.mean():.3f}  "
          f"min={max_p.min():.3f}  max={max_p.max():.3f}")

    mean_pol = pol_real.mean(axis=0)
    top5 = np.argsort(-mean_pol)[:5]
    print(f"  Top-5 actions by avg weight : {top5.tolist()}")
    print(f"  Their avg weights           : {np.round(mean_pol[top5], 4).tolist()}")
else:
    print("  No 'policies' key in file.")


# ── Per-session detail (first 40) ───────────────────────────────
sep("PER-SESSION DETAIL  (first 40)")
hdr = f"{'Sess':>5}  {'Len':>4}  {'Fill%':>6}  {'FinalDisc':>10}  {'Terminal':>9}  {'TgtMean':>8}  {'TgtMin':>7}  {'TgtMax':>7}"
if has_pol:
    hdr += f"  {'Visited':>8}  {'Entropy':>8}"
print(hdr)
for i in range(min(40, n_sess)):
    L   = int(ep_lens[i])
    m   = masks[i, :L]
    rs  = int(m.sum())
    fp  = f"{100*L/max_len:.0f}%"
    fd  = int(disc[i, L-1])
    is_t = bool(has_win[i] or has_loss[i])
    tm  = float(tgts[i, :L][m].mean()) if rs > 0 else 0.0
    tmi = float(tgts[i, :L][m].min())  if rs > 0 else 0.0
    tma = float(tgts[i, :L][m].max())  if rs > 0 else 0.0
    row = f"{i:>5}  {L:>4}  {fp:>6}  {fd:>10}  {'YES' if is_t else 'no':>9}  {tm:>8.3f}  {tmi:>7.3f}  {tma:>7.3f}"
    if has_pol and rs > 0:
        ep_pol = pols[i, :L][m]
        vis    = float(np.sum(ep_pol > 0, axis=-1).mean())
        ep_c   = np.clip(ep_pol, 1e-9, 1.0)
        ent    = float((-np.sum(ep_pol * np.log(ep_c), axis=-1)).mean())
        row   += f"  {vis:>8.1f}  {ent:>8.3f}"
    print(row)

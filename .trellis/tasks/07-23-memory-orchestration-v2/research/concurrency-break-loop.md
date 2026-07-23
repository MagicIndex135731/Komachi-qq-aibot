# Bug Analysis: late-arrival episode allocation races

## 1. Root Cause Category

- **Category**: D/E - test coverage gap and implicit assumption.
- **Specific cause**: allocation cached an open episode across transactions and
  assumed it remained current. Late-arrival resegmentation could supersede that
  episode between the read and membership insert.

## 2. Why Earlier Fixes Were Incomplete

1. Same-base generation compatibility fixed sequential late jobs, but did not
   cover a concurrent stale reader.
2. Reloading only the current message after a conflict could place a later
   message before memberships released by supersession. The entire unassigned
   batch has to be re-read and sorted again.

## 3. Prevention Mechanisms

| Priority | Mechanism | Specific action | Status |
|---|---|---|---|
| P0 | Architecture | Guard membership insert with current/open SQL CAS | DONE |
| P0 | Runtime | Restart the full chronological batch after CAS miss | DONE |
| P0 | Test | SQLite barrier test for read/supersede/write interleaving | DONE |
| P1 | Operations | Query superseded memberships and duplicate ordinals after deploy | DONE |
| P1 | Documentation | Record CAS and late-generation contracts in backend spec | DONE |

## 4. Systematic Expansion

- **Similar issues**: any read-decide-write path over versioned derived rows
  requires a write-time owner/generation/current predicate.
- **Design improvement**: repositories expose guarded state-transition methods;
  services treat CAS misses as expected concurrency rather than data errors.
- **Process improvement**: every destructive resegmentation test includes a
  paused competing worker, not only sequential calls.

## 5. Knowledge Capture

- [x] Backend executable contract updated.
- [x] Unit and integration assertions added.
- [x] Production database invariants checked after bot-only recreation.

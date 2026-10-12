# Paper outline: type-2 alternating codes

Working draft. Every number below comes from a file in this repository; the
file is named next to it. Placeholders are in square brackets.

**Working title:** Complete enumeration of binary type-2 alternating codes of
8 codes up to length [15], with a second length-14 code set

**Target venues:** *Radio Science* (where Sulzer 1993 and Virtanen et al. 2013
appeared) or *Annales Geophysicae* (Markkanen et al. 2008).

## Draft abstract

Type-2 alternating codes (Sulzer, 1993) give ambiguity-free incoherent scatter
autocorrelation measurements at code lengths that are not powers of two. Only
lengths 8–12 and 14 were known, and an efficient search method was lacking
(Markkanen et al., 2008). We present a meet-in-the-middle search that finds
every binary type-2 set of 8 codes of a given length, confirmed by an
independent implementation. We find 86, 78, 150, 118, 18, 0, and 4 sets for
lengths 8–14. The counts for lengths 8–10 reproduce Sulzer's. We show that no
set of 8 codes exists at length 13, and we report a second length-14 set that
is inequivalent to the published one. [Length 15 result.] We also identify
type-2 sets whose codes form complementary sets, which give a sidelobe-free
power profile from the same transmissions, and 4-phase type-2 sets of 4 codes
up to length 8.

## 1. Introduction

- Alternating codes: Lehtinen (1986); Lehtinen & Häggström (1987).
- Type-2 codes and their practical motivation, flexible pulse lengths: Sulzer
  (1993), abstract and discussion.
- State of knowledge:
  - Virtanen et al. (2013), *Radio Sci.* 48:519, para. [6]: type-2 codes "are
    only known for the lengths of 8, 9, 10, 11, 12 and 14 bits".
  - Markkanen et al. (2008), p. 2237: "there are no efficient search strategies
    for finding longer type 2 codes".
- Contributions: complete counts; nonexistence at 13; the new length-14 class;
  [length 15]; radar properties; polyphase extension.

## 2. Definitions and symmetries

- Type-2 condition and the equivalent fourth-order vector form:
  S[l, k] = Σ_codes Σ_j s_j s_{j+l} s_{j+k} s_{j+l+k} = 0.
- Per-code symmetries that leave a code's vector unchanged: negation,
  alternation, reversal.
- Global chirp symmetry: (−1)^(j(j−1)/2) multiplies S[l, k] by (−1)^(lk).
  Computed fact (N = 14): only 8 of the 2^14 global sign patterns preserve the
  condition (README, "Search for type-2 alternating codes").
- Counting convention: multisets of distinct vectors, which matches Sulzer
  (1993, p. 1000): "not all the same ACFs are used to make the set".

## 3. Method

- Linear 64-bit hash of the vectors, so there are no false negatives, with exact
  confirmation of every hit (`fastcodes/exhaust.py`).
- Independent clean-room C implementation (`cleanroom/t2search.c`), written
  from the definition alone, with an internal brute-force cross-check.
- Composition split (`exhaustive/t2abc.c`). The coordinates (1, N−3) and
  (2, N−3) depend only on the products c_j c_{j+N−3}, j = 0, 1, 2 (the
  argument in Sulzer 1993 about the last positions). Every code has one of
  three types, A = (2, 1), B = (0, −1) and C = (−2, 1), and a zero sum over
  8 codes forces exactly 2 A, 4 B and 2 C codes. The search matches
  (A-pair + C-pair) against B-multisets of size 4, which needs about 24 times
  less work than matching any 4 codes against any 4 codes (computed from the
  type counts, for example A = 528, B = 1024, C = 528 at N = 14).
  - Finer splits by more boundary coordinates don't reduce the work further
    (checked by counting at N = 12 and 13).
  - The program gives the same sets as `cleanroom/t2search` for N = 8–14.
- Validation:
  - Counts for N = 8–10 equal Sulzer's Table 3.
  - Sulzer's Table 4 sets for N = 8–12 are found (tests in `fastcodes/tests`).
  - The Markkanen et al. (2008) length-14 set is found.
- Computational cost: the work grows by about 16× per length. Length 14 took
  1,361 s with the clean-room program on 8 threads
  (`results/cleanroom/runlog14.txt`). [Add the `t2abc` timings for N = 14 and
  15 from a full 8-thread run.]

## 4. Results

**Table 1. Number of type-2 sets of 8 binary codes.** Sources:
`results/exhaust.jsonl` and `results/cleanroom/solutions_N*.txt`.

| N | Sets | Classes up to the chirp | Sulzer (1993) |
|---|---|---|---|
| 8 | 86 | 44 | 86 |
| 9 | 78 | 40 | 78 |
| 10 | 150 | 75 | 150 |
| 11 | 118 | 59 | 85 |
| 12 | 18 | 9 | 1 (partial) |
| 13 | 0 | 0 | — |
| 14 | 4 | 2 | — |
| 15 | [ ] | [ ] | — |

- **Length 11:** discuss 118 versus 85. Both of Sulzer's published sets appear in
  our lists. His Table 3 implies about 30 days of computing for N = 11.
- **New length-14 set:** 0C93 048D 02C7 0134 021E 0BF9 00F9 0454 (hex; bit 0 is
  +1, first element is the most significant bit). It is inequivalent to the
  published set because the vector norms differ.

## 5. Radar properties

- With matched decoding, every type-2 set with the same N and K has the same
  lag-profile noise level (derivation, plus numerical check).
- Power profile (zero lag) from the same transmissions:
  - The new length-14 set reaches a peak sidelobe of 2 and sidelobe energy 28,
    against 4 and 64 for the published set
    (`results/new_L14_best_power_profile.json`).
  - "Doubly complementary" sets: 68, 17, 24, 7 and 1 for N = 8–12
    (`results/type2_and_complementary.json`). A second script,
    `analysis/type1_overlap.py`, reproduces 68 for N = 8
    (`results/type1_overlap.txt`).
  - Overlap with type-1 codes at N = 8 (`results/type1_overlap.txt`): 8 of
    the 86 sets also meet Lehtinen's weak (type-1) condition for some choice
    of codes, and 5 of those 8 are doubly complementary. For N = 9–12 and 14, no 8-code type-2 set is
    type-1. Lehtinen's published 8-bit codes might include some of these
    sets; check this against Lehtinen (1986) and Lehtinen & Häggström (1987)
    before claiming N = 8.
  - Physics caveat. A complementary set cancels sidelobes only if the target
    stays coherent over the whole code cycle: Rastogi (1984), MAP Handbook 13,
    pp. 480–481, says that complementary code pairs "are not usable in
    mesospheric (incoherent-scatter) experiments with UHF radars"; Vierinen
    (2012, thesis), pp. 115–116, makes the same stationarity point. So the
    sidelobe-free power profile applies to coherent targets, not to
    incoherent scatter.
  - Standard practice doesn't get the zero lag from alternating-code
    decoding: Sulzer (1993), p. 996, "The zero lag is lost"; Lehtinen,
    Virtanen & Vierinen (2008), *Ann. Geophys.* 26:2291, Sec. 4.2. The closest
    practice is the "pseudo zero lag" from decoded power profiles in
    Häggström et al. (arXiv:physics/0003019v2, Sec. 2, pp. 3–4), with a
    16-bit strong code; they don't claim complementarity.
  - Turunen et al. (2002), *Ann. Geophys.* 20:1469, p. 1471, describe the
    gap: "more work is needed for finding a practical deterministic solution
    that contains all the wanted properties simultaneously."
- [Optional: the strong-condition extension, and receiver-filter effects.]

## 6. Polyphase extension

- Exact small searches (`results/polytype2_small.txt`):
  - 4-phase sets of 4 codes exist up to N = 8 (8 sets at N = 8, none at
    N = 9).
  - 3-phase sets of 6 codes exist at N = 5 and 7.
- Practical point: a 4-phase cycle of 4 codes, against 8 binary codes at N = 8.
- Prior work: no report of type-2 sets with fewer codes than the length was
  found. Markkanen et al. (2008), p. 2237, polyphase codes are type-1, with as
  many codes as the length. Vierinen et al. (2008), *Ann. Geophys.* 26:2923,
  Sec. 6, optimize short polyphase code groups numerically but list no codes
  and claim no exact type-2 property.

## 7. Discussion

- Why does N = 13 fail? This is open. Simple projection and residue arguments
  don't explain it.
- Beyond N = 15: cost, and the case of 12 codes.
  - Work of the composition split, left (A-pairs × C-pairs) plus right
    (B-multisets of size 4), as printed by `exhaustive/t2abc -n`:
    N = 14: 1.95e10 + 4.61e10; N = 15: 3.11e11 + 7.35e11;
    N = 16: 4.68e12 + 1.17e13; N = 17: 7.49e13 + 1.88e14. Each length costs
    about 16 times the previous one.
  - The same argument shows that a binary type-2 set needs a multiple of 4
    codes (K/4 A + K/2 B + K/4 C). Check how this relates to Sulzer (1993),
    p. 999, which discusses sets whose number of codes is a multiple of 4.
  - For 12 codes the composition is 3 A + 6 B + 3 C; a 6 + 6 split is too
    large for an exhaustive search, so 12 codes need the heuristic search
    (`fastcodes vsearch`, which now fixes the composition).

## Data and code availability

GitHub: kmbrgandhi/reu-code, branch `fast-search`.

## Open items before submission

- [ ] Length-15 run (`bash runs/resume.sh`; progress with `bash runs/progress.sh 15`).
- [ ] Close the literature gap: about 19 papers citing Sulzer that only Google
  Scholar lists, and NASA ADS. Both were blocked to automated access.
- [x] Novelty check of the doubly complementary and polyphase claims: no prior
  report found (agent notes, 2026-10-12). Still unread: Sulzer (1989),
  *Adv. Space Res.* 9(5):153 (cites Golay 1961 on complementary codes), and
  the full text of Lehtinen & Häggström (1987).
- [ ] Confirm with radar colleagues how power profiles are measured with
  alternating codes, before making the zero-lag claim.
- [ ] Figures: counts against N, and lag-profile and power-profile examples.
- [ ] Authorship and acknowledgments. Consider a co-author from MIT Haystack
  (Ryan Volz).
- [ ] Disclose AI assistance according to the journal's policy.

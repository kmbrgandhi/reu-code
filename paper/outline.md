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
- Validation:
  - Counts for N = 8–10 equal Sulzer's Table 3.
  - Sulzer's Table 4 sets for N = 8–12 are found (tests in `fastcodes/tests`).
  - The Markkanen et al. (2008) length-14 set is found.
- Computational cost: the work grows by about 16× per length. Length 14 took
  1,361 s with the C program (`results/cleanroom/runlog14.txt`).

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
    (`results/type2_and_complementary.json`).
- [Optional: the strong-condition extension, and receiver-filter effects.]

## 6. Polyphase extension

- Exact small searches (`results/polytype2_small.txt`):
  - 4-phase sets of 4 codes exist up to N = 8 (8 sets at N = 8, none at
    N = 9).
  - 3-phase sets of 6 codes exist at N = 5 and 7.
- Practical point: a 4-phase cycle of 4 codes, against 8 binary codes at N = 8.

## 7. Discussion

- Why does N = 13 fail? This is open. Simple projection and residue arguments
  don't explain it.
- Beyond N = 15: cost, and the case of 12 codes.

## Data and code availability

GitHub: kmbrgandhi/reu-code, branch `fast-search`.

## Open items before submission

- [ ] Length-15 run (`bash runs/resume.sh`; progress with `bash runs/progress.sh 15`).
- [ ] Close the literature gap: about 19 papers citing Sulzer that only Google
  Scholar lists, and NASA ADS. Both were blocked to automated access.
- [ ] Novelty check of the doubly complementary and polyphase claims (in progress).
- [ ] Confirm with radar colleagues how power profiles are measured with
  alternating codes, before making the zero-lag claim.
- [ ] Figures: counts against N, and lag-profile and power-profile examples.
- [ ] Authorship and acknowledgments. Consider a co-author from MIT Haystack
  (Ryan Volz).
- [ ] Disclose AI assistance according to the journal's policy.

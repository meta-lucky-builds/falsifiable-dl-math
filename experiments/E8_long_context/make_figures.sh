#!/usr/bin/env bash
# Draws the README figures from results_compact_toy.csv. Needs only awk.
#   bash experiments/E8_long_context/make_figures.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV="$HERE/results_compact_toy.csv"
OUT="$HERE/../../docs/figures"
mkdir -p "$OUT"

# Shared drawing code. Series colors are fixed by arm, so both figures agree.
COMMON='
function style() {
  print "<style>"
  print ".bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.grid{stroke:#e4e3df}.ref{stroke:#8a8984}"
  print ".alibi{stroke:#2a78d6}.rope_1e4{stroke:#eb6834}.rope_5e5{stroke:#1baf7a}.control_ctx4x{stroke:#eda100}"
  print "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.t1{fill:#f4f3ef}.t2{fill:#b5b4ae}.grid{stroke:#383835}.ref{stroke:#7c7b76}"
  print ".alibi{stroke:#3987e5}.rope_1e4{stroke:#d95926}.rope_5e5{stroke:#199e70}.control_ctx4x{stroke:#c98500}}"
  print "text{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;font-size:13px}"
  print ".title{font-size:16px;font-weight:600}.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
  print "</style>"
}
function label(x, y, s, cls, anchor) {
  printf "<text x=\"%.1f\" y=\"%.1f\" class=\"%s\" text-anchor=\"%s\">%s</text>\n", x, y, cls, anchor, s
}
function swatch(x, y, arm) {
  printf "<line x1=\"%.1f\" y1=\"%.1f\" x2=\"%.1f\" y2=\"%.1f\" class=\"line %s\"/>\n", x, y, x + 18, y, arm
}
'

# Figure 1: loss per row, mean over seeds.
awk -F, "$COMMON"'
BEGIN {
  W = 880; H = 430; L = 58; R = 150; T = 92; B = 56
  name["alibi"] = "ALiBi, trained on 256"; name["rope_1e4"] = "RoPE, trained on 256"
  name["rope_5e5"] = "RoPE, 50x theta, trained on 256"; name["control_ctx4x"] = "RoPE, trained on 1024"
  order[1] = "alibi"; order[2] = "rope_1e4"; order[3] = "rope_5e5"; order[4] = "control_ctx4x"
}
NR > 1 { n = split($8, v, "|"); for (i = 1; i <= n; i++) sum[$1, i] += v[i]; c[$1]++; bins = n }
function px(row) { return L + row / 2048 * (W - L - R) }
function py(b) { return H - B - b / 6 * (H - T - B) }
END {
  printf "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 %d %d\" role=\"img\" aria-label=\"Loss per row. RoPE models jump from about 1.5 to about 5 bits per byte just past their training length. ALiBi stays flat near 1.44.\">\n", W, H
  style()
  printf "<rect width=\"%d\" height=\"%d\" rx=\"8\" class=\"bg\"/>\n", W, H
  label(L, 28, "Loss stays flat with ALiBi, and RoPE breaks just past whatever length it trained on", "title t1", "start")
  xl = L
  for (k = 1; k <= 4; k++) {
    swatch(xl, 53, order[k]); label(xl + 24, 57, name[order[k]], "t2", "start")
    xl += 24 + length(name[order[k]]) * 6.7 + 20
  }
  for (b = 0; b <= 6; b++) {
    printf "<line x1=\"%d\" x2=\"%d\" y1=\"%.1f\" y2=\"%.1f\" class=\"grid\"/>\n", L, W - R, py(b), py(b)
    label(L - 10, py(b) + 4, b, "t2", "end")
  }
  for (r = 0; r <= 2048; r += 512) label(px(r), H - B + 20, r, "t2", "middle")
  label(L + (W - L - R) / 2, H - 14, "Position in the sequence (bytes of context available)", "t2", "middle")
  printf "<text transform=\"translate(16 %d) rotate(-90)\" class=\"t2\" text-anchor=\"middle\">Bits per byte</text>\n", T + (H - T - B) / 2
  split("256 1024", marks, " ")
  for (m = 1; m <= 2; m++) {
    printf "<line x1=\"%.1f\" x2=\"%.1f\" y1=\"%d\" y2=\"%d\" class=\"ref\" stroke-dasharray=\"4 4\"/>\n", px(marks[m]), px(marks[m]), T - 8, H - B
    label(px(marks[m]) + 5, T + 4, marks[m], "t2", "start")
  }
  for (k = 1; k <= 4; k++) {
    a = order[k]; d = ""
    for (i = 1; i <= bins; i++) d = d sprintf("%s%.1f,%.1f ", i == 1 ? "M" : "L", px(i * 2048 / bins - 32), py(sum[a, i] / c[a]))
    printf "<path d=\"%s\" class=\"line %s\"/>\n", d, a
    endv[a] = sum[a, bins] / c[a]
  }
  ex = px(2048) + 8
  label(ex, py(endv["rope_5e5"]) - 3, sprintf("RoPE 50x %.2f", endv["rope_5e5"]), "t1", "start")
  label(ex, py(endv["rope_1e4"]) + 13, sprintf("RoPE %.2f", endv["rope_1e4"]), "t1", "start")
  label(ex, py(endv["control_ctx4x"]) + 4, sprintf("RoPE 1024 %.2f", endv["control_ctx4x"]), "t1", "start")
  label(ex, py(endv["alibi"]) + 4, sprintf("ALiBi %.2f", endv["alibi"]), "t1", "start")
  print "</svg>"
}' "$CSV" > "$OUT/e8_loss_per_row.svg"

# Figure 2: copy probe, mean over seeds, log distance axis.
awk -F, "$COMMON"'
BEGIN {
  W = 880; H = 400; L = 62; R = 150; T = 92; B = 56
  split("32 64 128 200 400 800 1600", D, " ")
  name["alibi"] = "ALiBi, trained on 256"; name["rope_1e4"] = "RoPE, trained on 256"
  name["rope_5e5"] = "RoPE, 50x theta, trained on 256"; name["control_ctx4x"] = "RoPE, trained on 1024"
  order[1] = "alibi"; order[2] = "rope_1e4"; order[3] = "rope_5e5"; order[4] = "control_ctx4x"
}
NR > 1 { n = split($11, v, "|"); for (i = 1; i <= n; i++) sum[$1, i] += v[i]; c[$1]++; pts = n }
function px(d) { return L + log(d / 32) / log(1600 / 32) * (W - L - R) }
function py(a) { return H - B - a / 0.3 * (H - T - B) }
END {
  printf "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 %d %d\" role=\"img\" aria-label=\"Copy accuracy against distance. At distance 400, ALiBi trained on 256 still recovers bytes at about 0.135, while RoPE trained on 256 is at chance.\">\n", W, H
  style()
  printf "<rect width=\"%d\" height=\"%d\" rx=\"8\" class=\"bg\"/>\n", W, H
  label(L, 28, "ALiBi still recovers bytes of a planted string at distances it never trained on", "title t1", "start")
  xl = L
  for (k = 1; k <= 4; k++) {
    swatch(xl, 53, order[k]); label(xl + 24, 57, name[order[k]], "t2", "start")
    xl += 24 + length(name[order[k]]) * 6.7 + 20
  }
  for (a = 0; a <= 0.3001; a += 0.1) {
    printf "<line x1=\"%d\" x2=\"%d\" y1=\"%.1f\" y2=\"%.1f\" class=\"grid\"/>\n", L, W - R, py(a), py(a)
    label(L - 10, py(a) + 4, sprintf("%.1f", a), "t2", "end")
  }
  for (i = 1; i <= 7; i++) label(px(D[i]), H - B + 20, D[i], "t2", "middle")
  label(L + (W - L - R) / 2, H - 14, "Distance back to the planted string, in bytes (log scale)", "t2", "middle")
  printf "<text transform=\"translate(16 %d) rotate(-90)\" class=\"t2\" text-anchor=\"middle\">Byte accuracy</text>\n", T + (H - T - B) / 2
  printf "<line x1=\"%d\" x2=\"%d\" y1=\"%.1f\" y2=\"%.1f\" class=\"ref\" stroke-dasharray=\"4 4\"/>\n", L, W - R, py(1 / 62), py(1 / 62)
  label(W - R + 8, py(1 / 62) + 4, "Guessing, 1/62", "t2", "start")
  printf "<line x1=\"%.1f\" x2=\"%.1f\" y1=\"%d\" y2=\"%d\" class=\"ref\" stroke-dasharray=\"4 4\"/>\n", px(256), px(256), T - 8, H - B
  label(px(256) + 5, T + 4, "256", "t2", "start")
  for (k = 1; k <= 4; k++) {
    a = order[k]; d = ""
    for (i = 1; i <= pts; i++) d = d sprintf("%s%.1f,%.1f ", i == 1 ? "M" : "L", px(D[i]), py(sum[a, i] / c[a]))
    printf "<path d=\"%s\" class=\"line %s\"/>\n", d, a
    for (i = 1; i <= pts; i++) printf "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"4\" class=\"line %s bg\"/>\n", px(D[i]), py(sum[a, i] / c[a]), a
  }
  print "</svg>"
}' "$CSV" > "$OUT/e8_copy_probe.svg"
echo "wrote $OUT"

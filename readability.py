"""
Measure sentence length in the manuscript, old version against new.

The reviewer's first objection was that sentences were hard to follow. Sentence
length is a crude but honest proxy: long sentences carrying several clauses are the
ones that force a reader back to the start. This compares the committed version that
was rejected against the current text.

Run: python readability.py
"""
import io
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD_COMMIT = "6ba81b8"
SECTIONS = ["introduction", "related_work", "signal_model", "threat_model",
            "methods", "results", "discussion", "conclusion"]


def strip_tex(t):
    t = re.sub(r"%.*", " ", t)                        # comments
    t = re.sub(r"\\begin\{(table|tabular|figure|equation)\*?\}.*?"
               r"\\end\{\1\*?\}", " ", t, flags=re.S)  # floats and maths
    t = re.sub(r"\\cite[tp]?\*?(\[[^\]]*\])*\{[^}]*\}", "CITE", t)
    t = re.sub(r"\\(ref|eqref)\{[^}]*\}", "REF", t)
    t = re.sub(r"\$[^$]*\$", "MATH", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
    t = re.sub(r"[{}&\\]", " ", t)
    return re.sub(r"\s+", " ", t)


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) > 3]


def stats(text):
    ss = sentences(strip_tex(text))
    if not ss:
        return None
    lens = [len(s.split()) for s in ss]
    lens.sort()
    long40 = sum(1 for x in lens if x > 40)
    return dict(n=len(lens), mean=sum(lens) / len(lens),
                median=lens[len(lens) // 2], longest=lens[-1],
                over40=long40, pct40=100 * long40 / len(lens))


def main():
    new = "".join(io.open(HERE / "sections" / f"{s}.tex", encoding="utf-8").read()
                  for s in SECTIONS if (HERE / "sections" / f"{s}.tex").exists())
    old = ""
    for s in SECTIONS:
        r = subprocess.run(["git", "-C", str(HERE), "show",
                            f"{OLD_COMMIT}:sections/{s}.tex"],
                           capture_output=True, text=True, encoding="utf-8")
        old += r.stdout or ""

    print(f"{'':10s}{'sentences':>10s}{'mean':>8s}{'median':>8s}"
          f"{'longest':>9s}{'>40 words':>11s}")
    for name, txt in (("rejected", old), ("current", new)):
        st = stats(txt)
        if not st:
            print(f"  {name}: no text"); continue
        print(f"{name:10s}{st['n']:>10d}{st['mean']:>8.1f}{st['median']:>8d}"
              f"{st['longest']:>9d}{st['over40']:>7d} ({st['pct40']:.1f}%)")


if __name__ == "__main__":
    main()

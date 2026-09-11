"""
Check every citation, label and cross reference in the manuscript.

Run after editing any section:  python check_refs.py
Reports citations with no bib entry, references with no label, and labels never used.
"""
import re
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = [HERE / "main.tex"] + sorted((HERE / "sections").glob("*.tex"))

CITE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]*)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
BIBKEY = re.compile(r"@[a-zA-Z]+\{([^,]+),")


def main():
    text = ""
    for p in TEX:
        if p.exists():
            text += io.open(p, encoding="utf-8", errors="ignore").read() + "\n"

    cites, refs, labels = set(), set(), set()
    for m in CITE.finditer(text):
        cites.update(k.strip() for k in m.group(1).split(",") if k.strip())
    for m in REF.finditer(text):
        refs.add(m.group(1).strip())
    for m in LABEL.finditer(text):
        labels.add(m.group(1).strip())

    bib = io.open(HERE / "references.bib", encoding="utf-8", errors="ignore").read()
    have = set(BIBKEY.findall(bib))

    missing_cites = sorted(c for c in cites if c not in have)
    missing_refs = sorted(r for r in refs if r not in labels)
    unused_labels = sorted(l for l in labels if l not in refs)

    print(f"citations used      : {len(cites)}")
    print(f"  missing from .bib : {missing_cites if missing_cites else 'none'}")
    print(f"cross references    : {len(refs)}")
    print(f"  with no label     : {missing_refs if missing_refs else 'none'}")
    print(f"labels defined      : {len(labels)}")
    print(f"  never referenced  : {unused_labels if unused_labels else 'none'}")
    ok = not missing_cites and not missing_refs
    print("\nSTATUS:", "OK" if ok else "BROKEN REFERENCES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Rebuild the Satellite Navigation submission package from the live source.

The package is a flat directory: one manuscript.tex with the sections inlined,
the figures beside it rather than in a subdirectory, and the class, style and
bibliography files the manuscript needs. Nothing here is edited by hand. Edit
main.tex and sections/, then rerun this.
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "submission")
BS = chr(92)

KEEP = ("cover_letter.tex", "cover_letter.pdf", "title_page.tex", "title_page.pdf")
SUPPORT = ("sn-jnl.cls", "sn-basic.bst", "cuted.sty", "references.bib")


def inline(main_text):
    """Replace each \\input{sections/x} with the file's content."""
    def repl(m):
        p = os.path.join(HERE, m.group(1) + ".tex")
        if not os.path.exists(p):
            sys.exit("missing input: " + p)
        return open(p, encoding="utf-8").read().rstrip() + "\n"
    return re.sub(re.escape(BS) + r"input\{([^}]*)\}", repl, main_text)


def main():
    os.makedirs(SUB, exist_ok=True)

    # Remove only what this script generates. Hand written files are kept.
    for f in os.listdir(SUB):
        if f in KEEP:
            continue
        p = os.path.join(SUB, f)
        if os.path.isfile(p):
            os.remove(p)

    text = inline(open(os.path.join(HERE, "main.tex"), encoding="utf-8").read())

    # Figures sit beside the manuscript in the package, not under figures/.
    text = text.replace("{figures/", "{")
    # Internal notes to ourselves are not part of the submission.
    text = re.sub(r"^%.*(?:Figure source|figures_src|sync\.py|Do not edit).*\n", "",
                  text, flags=re.M)

    out = os.path.join(SUB, "manuscript.tex")
    open(out, "w", encoding="utf-8").write(text)

    used = set(re.findall(re.escape(BS) + r"includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text))
    figdir = os.path.join(HERE, "figures")
    missing = []
    for f in sorted(used):
        src = os.path.join(figdir, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SUB, os.path.basename(f)))
        else:
            missing.append(f)
    if missing:
        sys.exit("figures not found: " + ", ".join(missing))

    for f in SUPPORT:
        src = os.path.join(HERE, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SUB, f))
        else:
            print("  note: %s not beside main.tex, not copied" % f)

    print("manuscript.tex written, %d figures, %d support files" % (len(used), len(SUPPORT)))

    for i in range(3):
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "manuscript.tex"],
                       cwd=SUB, capture_output=True, text=True)
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "manuscript.tex"],
                       cwd=SUB, capture_output=True, text=True)
    log = os.path.join(SUB, "manuscript.log")
    bad = []
    if os.path.exists(log):
        lt = open(log, encoding="utf-8", errors="replace").read()
        bad = sorted(set(re.findall(r"(?:Citation|Reference) `([^']+)' .*undefined", lt)))
    print("latexmk exit %d; undefined: %s" % (r.returncode, bad if bad else "none"))

    # A submission carries sources and the built PDF, not build artefacts.
    for f in os.listdir(SUB):
        if os.path.splitext(f)[1] in (".aux", ".log", ".blg", ".out", ".fls",
                                      ".fdb_latexmk", ".synctex.gz", ".xdv"):
            os.remove(os.path.join(SUB, f))
    print("package: %d files" % len(os.listdir(SUB)))


if __name__ == "__main__":
    main()

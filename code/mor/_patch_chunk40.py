import os, py_compile
path = "03_code/p1/p2_grid_self.py"
s = open(path, encoding="utf-8").read()

old1 = (
    "ap = argparse.ArgumentParser()\n"
    "ap.add_argument(\"--count\", type=int, default=400)\n"
    "ap.add_argument(\"--merge\", action=\"store_true\")\n"
    "args = ap.parse_args()\n"
    "\n"
    "if args.merge:\n"
    "    ch = np.load(os.path.join(OUT, \"p2_grid_self_c0.npz\"), allow_pickle=True)\n"
    "    print(\"npz keys:\", list(ch.files), flush=True)\n"
    "    sys.exit(0)\n"
    "\n"
)
new1 = (
    "ap = argparse.ArgumentParser()\n"
    "ap.add_argument(\"--start\", type=int, default=0)\n"
    "ap.add_argument(\"--count\", type=int, default=40)\n"
    "ap.add_argument(\"--merge\", action=\"store_true\")\n"
    "args = ap.parse_args()\n"
    "\n"
)
assert old1 in s, "old1 not found"
s = s.replace(old1, new1, 1)

s = s.replace("sel_chunk = sel[:args.count]",
              "sel_chunk = sel[args.start:args.start+args.count]", 1)

# union mask: build over the chunk's own airfoils (reproducible per-chunk), not all sel
old_mask_loop = "for sim in sel:\n    mm = pv.read(os.path.join(DATASET, sim, sim + \"_aerofoil.vtp\"))"
new_mask_loop = "for sim in sel_chunk:\n    mm = pv.read(os.path.join(DATASET, sim, sim + \"_aerofoil.vtp\"))"
if old_mask_loop in s:
    s = s.replace(old_mask_loop, new_mask_loop, 1)

open(path, "w", encoding="utf-8").write(s)
py_compile.compile(path, doraise=True)
print("patched p2_grid_self.py to start/count chunking; compile OK")

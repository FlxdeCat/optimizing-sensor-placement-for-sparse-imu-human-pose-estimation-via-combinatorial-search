"""Run only in your trusted legacy SMPL environment after editing the two paths.

No Chumpy is needed in the main project. This tool cannot make an incompatible
model into 24-joint SMPL; 0_check.py validates the resulting field shapes.
"""
from pathlib import Path
import pickle
import numpy as np
SOURCE=Path("SMPL_male.pkl")
OUTPUT=Path("SMPL_male.npz")
FIELDS=("J_regressor","weights","posedirs","shapedirs","v_template","J","f","kintree_table")


def main():
    if not SOURCE.is_file():raise RuntimeError("Edit SOURCE to your licensed SMPL pickle in this script.")
    if OUTPUT.exists():raise RuntimeError("OUTPUT exists; choose an unused output path.")
    with SOURCE.open("rb") as stream:body=pickle.load(stream,encoding="latin1")
    arrays={}
    for key in FIELDS:
        value=body[key]
        if hasattr(value,"toarray"):value=value.toarray()
        if hasattr(value,"r"):value=value.r
        arrays[key]=np.asarray(value)
        if arrays[key].dtype.kind not in "fiu" or not np.isfinite(arrays[key]).all():raise ValueError(f"Invalid numeric SMPL field {key}")
    OUTPUT.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUTPUT,**arrays)
    print("Numeric SMPL written:",OUTPUT.resolve())


if __name__=="__main__":main()

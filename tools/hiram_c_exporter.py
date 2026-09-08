import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

def main():
    eval_c = (REPO_ROOT / "src/hiram_eval.c").read_bytes()
    eval_h = (REPO_ROOT / "include/hiram/hiram_eval.h").read_bytes()
    circ_h = (REPO_ROOT / "include/hiram/hiram_circuit_data.h").read_bytes()

    pathlib.Path("hiram_eval.c").write_bytes(eval_c)
    pathlib.Path("hiram_eval.h").write_bytes(eval_h)
    pathlib.Path("hiram_circuit_data.h").write_bytes(circ_h)

if __name__ == "__main__":
    main()
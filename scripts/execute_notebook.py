import ast
import base64
import contextlib
import io
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "exploratory_analysis.ipynb"


def run_notebook():
    print(f"Reading notebook from: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    os.chdir(ROOT / "notebooks")
    globs = {"__name__": "__main__"}
    exec_count = 1

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source.strip()
        if not source:
            continue

        print(f"Executing code cell {exec_count}...")
        captured_outputs = []

        # Hook plt.show to capture figures
        def custom_show(*args, **kwargs):
            fig = plt.gcf()
            if fig.get_axes():
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("ascii")
                captured_outputs.append(
                    nbformat.v4.new_output(
                        output_type="display_data",
                        data={
                            "image/png": b64,
                            "text/plain": f"<Figure size {fig.get_size_inches()[0]*110:.0f}x{fig.get_size_inches()[1]*110:.0f} with {len(fig.get_axes())} Axes>",
                        },
                        metadata={},
                    )
                )
            plt.close(fig)

        plt.show = custom_show

        stdout_buf = io.StringIO()
        try:
            tree = ast.parse(source)
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = tree.body.pop()

            with contextlib.redirect_stdout(stdout_buf):
                if tree.body:
                    code_to_exec = compile(tree, filename="<cell>", mode="exec")
                    exec(code_to_exec, globs)
                if last_expr is not None:
                    code_to_eval = compile(
                        ast.Expression(body=last_expr.value),
                        filename="<cell>",
                        mode="eval",
                    )
                    res = eval(code_to_eval, globs)
                    if res is not None:
                        data = {"text/plain": repr(res)}
                        if hasattr(res, "_repr_html_"):
                            html = res._repr_html_()
                            if html:
                                data["text/html"] = html
                        captured_outputs.append(
                            nbformat.v4.new_output(
                                output_type="execute_result",
                                data=data,
                                execution_count=exec_count,
                                metadata={},
                            )
                        )
        except Exception as e:
            print(f"Error executing cell {exec_count}: {e}")
            raise

        # Check if any remaining plot wasn't shown
        if plt.get_fignums():
            custom_show()

        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            captured_outputs.insert(
                0,
                nbformat.v4.new_output(
                    output_type="stream",
                    name="stdout",
                    text=stdout_text,
                ),
            )

        cell.outputs = captured_outputs
        cell.execution_count = exec_count
        exec_count += 1

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Successfully executed {exec_count - 1} code cells and updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    run_notebook()

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors import CellExecutionError

import re

def test_notebook():
    notebook_filename = 'test.ipynb'

    # Load the notebook
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        # Preprocess notebook, execute all cells
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except CellExecutionError as e:
        # If there is any error (including assertion failure) in any cell, raise it
        raise AssertionError(f"Error executing the notebook '{notebook_filename}':\n{e}")

    # Optionally, you can check outputs of specific cells if needed
    # for example, to check an output of cell 5 you could use:
    print(nb.cells[3].outputs)
    cell_output = nb.cells[3].outputs[0].text
    assert re.search('Epoch 1:', cell_output)

    # If no exceptions were raised, it means all assertions in the notebook passed


if __name__ == "__main__":
    test_notebook()
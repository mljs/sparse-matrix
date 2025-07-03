import { SparseMatrix } from '../index.js';

/**
 * Multiplies two very small sparse matrices (both with cardinalities < 42).
 *
 * @private
 * @param {SparseMatrix} left - The left-hand side matrix.
 * @param {SparseMatrix} right - The right-hand side matrix to multiply with.
 * @returns {SparseMatrix} The resulting matrix after multiplication.
 */
export function mmulSmall(left, right) {
  const {
    columns: otherCols,
    rows: otherRows,
    values: otherValues,
  } = right.getNonZeros();

  const nbOtherActive = otherCols.length;

  const m = left.rows;
  const p = right.columns;
  const output = new SparseMatrix(m, p);

  left.withEachNonZero((i, j, v1) => {
    for (let o = 0; o < nbOtherActive; o++) {
      if (j === otherRows[o]) {
        const l = otherCols[o];
        output.set(i, l, output.get(i, l) + otherValues[o] * v1);
      }
    }
  });
  return output;
}

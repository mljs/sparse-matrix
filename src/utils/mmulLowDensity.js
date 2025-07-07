import { SparseMatrix } from '../index.js';

/**
 * Multiplies two sparse matrices, optimized for cases where the right-hand side matrix
 * has low density (number of rows > 100 and cardinality < 100).
 *
 * @private
 * @param {SparseMatrix} left - The left-hand side matrix.
 * @param {SparseMatrix} right - The right-hand side matrix to multiply with.
 * @returns {SparseMatrix} The resulting matrix after multiplication.
 */
export function mmulLowDensity(left, right) {
  const {
    columns: otherCols,
    rows: otherRows,
    values: otherValues,
  } = right.getNonZeros();
  const {
    columns: thisCols,
    rows: thisRows,
    values: thisValues,
  } = left.getNonZeros();

  const m = left.rows;
  const p = right.columns;
  const output = new SparseMatrix(m, p);

  const nbOtherActive = otherCols.length;
  const nbThisActive = thisCols.length;
  for (let t = 0; t < nbThisActive; t++) {
    const i = thisRows[t];
    const j = thisCols[t];
    for (let o = 0; o < nbOtherActive; o++) {
      if (j === otherRows[o]) {
        const l = otherCols[o];
        output.set(i, l, output.get(i, l) + otherValues[o] * thisValues[t]);
      }
    }
  }
  return output;
}

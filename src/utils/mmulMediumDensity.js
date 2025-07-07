import { SparseMatrix } from '../index.js';

/**
 * Multiplies two sparse matrices where the right-hand side matrix is of medium density.
 * Uses CSR format for the right-hand side for efficient multiplication.
 *
 * @private
 * @param {SparseMatrix} left - The left-hand side matrix.
 * @param {SparseMatrix} right - The right-hand side matrix to multiply with (in CSR format).
 * @returns {SparseMatrix} The resulting matrix after multiplication.
 */

export function mmulMediumDensity(left, right) {
  const m = left.rows;
  const p = right.columns;
  const result = new SparseMatrix(m, p);
  const {
    columns: thisCols,
    rows: thisRows,
    values: thisValues,
  } = left.getNonZeros();
  const {
    columns: otherCols,
    rows: otherRows,
    values: otherValues,
  } = right.getNonZeros({ csr: true });

  const nbThisActive = thisCols.length;
  for (let t = 0; t < nbThisActive; t++) {
    const i = thisRows[t];
    const j = thisCols[t];
    const oStart = otherRows[j];
    const oEnd = otherRows[j + 1];
    for (let k = oStart; k < oEnd; k++) {
      const l = otherCols[k];
      result.set(i, l, result.get(i, l) + otherValues[k] * thisValues[t]);
    }
  }

  return result;
}

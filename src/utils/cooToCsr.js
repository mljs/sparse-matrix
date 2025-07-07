/**
 * Converts a matrix from Coordinate (COO) format to Compressed Sparse Row (CSR) format.
 * @param {Object} cooMatrix - The matrix in COO format with properties: values, columns, rows.
 * @param {number} nbRows - The number of rows in the matrix.
 * @returns {{rows: Float64Array, columns: Float64Array, values: Float64Array}} The matrix in CSR format.
 */
export function cooToCsr(cooMatrix, nbRows) {
  const { values, columns, rows } = cooMatrix;
  const csrRowPtr = new Float64Array(nbRows + 1);

  // Count non-zeros per row
  const numberOfNonZeros = rows.length;
  for (let i = 0; i < numberOfNonZeros; i++) {
    csrRowPtr[rows[i] + 1]++;
  }

  // Compute cumulative sum
  for (let i = 1; i <= nbRows; i++) {
    csrRowPtr[i] += csrRowPtr[i - 1];
  }

  return { rows: csrRowPtr, columns, values };
}

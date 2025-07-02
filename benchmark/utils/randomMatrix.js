export function randomMatrix(rows, cols, cardinality) {
  const total = rows * cols;
  const positions = new Set();

  // Generate unique random positions
  while (positions.size < cardinality) {
    positions.add(Math.floor(Math.random() * total));
  }

  // Build the matrix with zeros
  const matrix = Array.from({ length: rows }, () => new Float64Array(cols));

  // Assign random values to the selected positions
  for (const pos of positions) {
    const i = Math.floor(pos / cols);
    const j = pos % cols;
    matrix[i][j] = Math.random() * 10;
  }

  return matrix;
}

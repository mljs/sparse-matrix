import { Matrix } from 'ml-matrix';
import { describe, expect, it } from 'vitest';

import { SparseMatrix } from '../index.js';

describe('Sparse Matrix', () => {
  it('add', () => {
    let m1 = new SparseMatrix([
      [2, 0, 1],
      [0, 0, 3],
      [2, 0, 1],
    ]);
    let m2 = new SparseMatrix([
      [0, 1, 5],
      [2, 0, 0],
      [-2, 0, -1],
    ]);
    let m3 = m1.add(m2).to2DArray();
    expect(m3).toStrictEqual([
      [2, 1, 6],
      [2, 0, 3],
      [0, 0, 0],
    ]);
  });
  it('mmul', () => {
    let m1 = new SparseMatrix([
      [2, 0, 1],
      [0, 0, 3],
    ]);
    let m2 = new SparseMatrix([
      [0, 1],
      [2, 0],
      [0, 0],
    ]);
    let m3 = m1.mmul(m2);

    expect(m1.cardinality).toBe(3);
    expect(m2.cardinality).toBe(2);
    expect(m3.cardinality).toBe(1);

    expect(m3.get(0, 1)).toBe(2);
    expectMatrixClose(m3.to2DArray(), [
      [0, 2],
      [0, 0],
    ]);

    // Compare with dense multiplication
    const denseM1 = new Matrix(m1.to2DArray());
    const denseM2 = new Matrix(m2.to2DArray());
    const expectedDense = denseM1.mmul(denseM2);
    expectMatrixClose(m3.to2DArray(), expectedDense.to2DArray());
  });

  it('mmul', () => {
    const size = 32;
    const density = 0.1;
    const A = randomMatrix(size, size, density * size ** 2);
    const B = randomMatrix(size, size, density * size ** 2);
    const m1 = new SparseMatrix(A);
    const m2 = new SparseMatrix(B);
    const m3 = m1.mmul(m2);

    const denseM1 = new Matrix(A);
    const denseM2 = new Matrix(B);
    const expectedDense = denseM1.mmul(denseM2);
    expectMatrixClose(m3.to2DArray(), expectedDense.to2DArray());
  });

  it('mmul with low density', () => {
    const size = 128;
    const cardinality = 64;
    const A = randomMatrix(size, size, cardinality);
    const B = randomMatrix(size, size, cardinality);
    const m1 = new SparseMatrix(A);
    const m2 = new SparseMatrix(B);
    const m3 = m1.mmul(m2);

    const denseM1 = new Matrix(A);
    const denseM2 = new Matrix(B);
    const expectedDense = denseM1.mmul(denseM2);
    expectMatrixClose(m3.to2DArray(), expectedDense.to2DArray());
  });

  it('kronecker', () => {
    const matrix1 = new SparseMatrix([
      [1, 2],
      [3, 4],
    ]);
    const matrix2 = new SparseMatrix([
      [0, 5],
      [6, 7],
    ]);
    const product = matrix1.kroneckerProduct(matrix2);
    expect(product.to2DArray()).toStrictEqual([
      [0, 5, 0, 10],
      [6, 7, 12, 14],
      [0, 15, 0, 20],
      [18, 21, 24, 28],
    ]);
  });

  it('isSymmetric', () => {
    expect(new SparseMatrix(10, 10).isSymmetric()).toBe(true);
    expect(new SparseMatrix(15, 10).isSymmetric()).toBe(false);

    let m = new SparseMatrix([
      [0, 1],
      [1, 0],
    ]);
    expect(m.isSymmetric()).toBe(true);

    m = new SparseMatrix([
      [0, 1],
      [0, 1],
    ]);
    expect(m.isSymmetric()).toBe(false);
  });

  it('transpose', () => {
    const matrix = new SparseMatrix([
      [1, 2],
      [3, 4],
    ]);
    expect(matrix.transpose().to2DArray()).toStrictEqual([
      [1, 3],
      [2, 4],
    ]);
  });
});

describe('Banded matrices', () => {
  it('Check band size', () => {
    const matrix1 = new SparseMatrix([
      [1, 0],
      [0, 1],
    ]);
    const matrix2 = new SparseMatrix([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const matrix3 = new SparseMatrix([
      [1, 0, 1],
      [0, 1, 0],
    ]);
    const matrix4 = new SparseMatrix([
      [1, 0, 0],
      [1, 1, 0],
    ]);
    const matrix5 = new SparseMatrix([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
    ]);
    expect(matrix1.bandWidth()).toBe(0);
    expect(matrix2.bandWidth()).toBe(0);
    expect(matrix3.bandWidth()).toBe(2);
    expect(matrix4.bandWidth()).toBe(1);
    expect(matrix5.bandWidth()).toBe(0);
  });

  it('isBanded', () => {
    const matrix1 = new SparseMatrix([
      [1, 0],
      [0, 1],
    ]);
    const matrix2 = new SparseMatrix([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const matrix3 = new SparseMatrix([
      [1, 0, 1],
      [0, 1, 0],
    ]);
    const matrix4 = new SparseMatrix([
      [1, 0, 0],
      [1, 1, 0],
    ]);
    expect(matrix1.isBanded(1)).toBe(true);
    expect(matrix2.isBanded(1)).toBe(true);
    expect(matrix3.isBanded(1)).toBe(false);
    expect(matrix4.isBanded(1)).toBe(true);
  });
});

/**
 * Helper to compare two 2D arrays element-wise using toBeCloseTo.
 */
function expectMatrixClose(received, expected, precision = 6) {
  expect(received.length).toBe(expected.length);
  for (let i = 0; i < received.length; i++) {
    expect(received[i].length).toBe(expected[i].length);
    for (let j = 0; j < received[i].length; j++) {
      expect(received[i][j]).toBeCloseTo(expected[i][j], precision);
    }
  }
}

function randomMatrix(rows, cols, cardinality) {
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

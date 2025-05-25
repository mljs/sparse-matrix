import { describe, expect, it } from 'vitest';

import { SparseMatrix } from '../index.js';

describe('Sparse Matrix', () => {
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

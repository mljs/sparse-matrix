import { describe, expect, it } from 'vitest';

import { SparseMatrix } from '../index.js';

describe('Sparse Matrix', () => {
  it('getNonZeros', () => {
    let m2 = new SparseMatrix(
      [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 2, 1, 1],
        [0, 3, 0, 0, 5, 5],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 9, 9],
      ],
      { initialCapacity: 12 },
    );

    // Default (coordinate list)
    expect(m2.getNonZeros()).toEqual({
      rows: Float64Array.from([4, 1, 4, 1, 1, 1, 4, 2, 4, 2, 2]),
      columns: Float64Array.from([0, 0, 3, 3, 4, 5, 5, 1, 4, 4, 5]),
      values: Float64Array.from([1, 1, 1, 2, 1, 1, 9, 3, 9, 5, 5]),
    });

    // CSR format
    expect(m2.getNonZeros({ format: 'csr' })).toEqual({
      rows: Float64Array.from([0, 0, 4, 7, 7, 11]),
      columns: Float64Array.from([0, 3, 4, 5, 1, 4, 5, 0, 3, 4, 5]),
      values: Float64Array.from([1, 2, 1, 1, 3, 5, 5, 1, 1, 9, 9]),
    });

    //CSC format
    expect(m2.getNonZeros({ format: 'csc' })).toEqual({
      rows: Float64Array.from([1, 4, 2, 1, 4, 1, 2, 4, 1, 2, 4]),
      columns: Float64Array.from([0, 2, 3, 3, 5, 8, 11]),
      values: Float64Array.from([1, 1, 3, 2, 1, 1, 5, 9, 1, 5, 9]),
    });
  });
});

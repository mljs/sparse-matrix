import { run, bench, lineplot, do_not_optimize } from 'mitata';
import { SparseMatrix } from '../src/index.js';
import { randomSparseMatrix } from './utils/randomSparseMatrix.js';
import { Matrix } from 'ml-matrix';
import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld.js';
import { randomMatrix } from './utils/randomMatrix.js';
const density = 0.01; // Fixed density for this comparison;
const min = 32;
const max = 128;
// Prepare matrices once

lineplot(() => {
  bench('Sparse.mmul($size)', function* (ctx) {
    const size = ctx.get('size');

    // Prepare matrices once
    const A = new SparseMatrix(randomMatrix(size, size, density));
    const B = new SparseMatrix(randomMatrix(size, size, density));
    // Benchmark the multiplication
    yield () => do_not_optimize(A.mmul(B));
  }).range('size', min, max, 2); // 16, 32, 64, 128, 256

  bench('SparseOld.mmul($size)', function* (ctx) {
    const size = ctx.get('size');
    const A = randomMatrix(size, size, density);
    const B = randomMatrix(size, size, density);
    const AOld = new SparseMatrixOld(A);
    const BOld = new SparseMatrixOld(B);

    // Benchmark the multiplication
    yield () => do_not_optimize(AOld.mmul(BOld));
  }).range('size', min, max, 2);

  bench('Dense.mmul($size)', function* (ctx) {
    const size = ctx.get('size');

    // Prepare matrices once
    const A = randomMatrix(size, size, density);
    const B = randomMatrix(size, size, density);
    const ADense = new Matrix(A);
    const BDense = new Matrix(B);

    // Benchmark the multiplication
    yield () => do_not_optimize(ADense.mmul(BDense));
  }).range('size', min, max, 2);
});

await run();

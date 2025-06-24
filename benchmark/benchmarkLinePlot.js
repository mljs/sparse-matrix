import { run, bench, lineplot, do_not_optimize } from 'mitata';
import { SparseMatrix } from '../src/index.js';
import { xSequentialFillFromStep } from 'ml-spectra-processing';
import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld.js';
import { randomMatrix } from './utils/randomMatrix.js';
const density = 0.02; // Fixed density for this comparison;

// Prepare matrices once
const sizes = Array.from(
  xSequentialFillFromStep({ from: 4, step: 4, size: 13 }),
);
lineplot(() => {
  bench('Sparse.mmul($size)', function* (ctx) {
    const size = ctx.get('size');

    // Prepare matrices once
    const A = new SparseMatrix(randomMatrix(size, size, density));
    const B = new SparseMatrix(randomMatrix(size, size, density));
    // Benchmark the multiplication
    yield () => do_not_optimize(A.mmul(B));
  }).args('size', sizes); // 16, 32, 64, 128, 256

  bench('SparseOld.mmul($size)', function* (ctx) {
    const size = ctx.get('size');
    const A = randomMatrix(size, size, density);
    const B = randomMatrix(size, size, density);
    const AOld = new SparseMatrixOld(A);
    const BOld = new SparseMatrixOld(B);

    // Benchmark the multiplication
    yield () => do_not_optimize(AOld.mmul(BOld));
  }).args('size', sizes);

  // bench('Dense.mmul($size)', function* (ctx) {
  //   const size = ctx.get('size');

  //   // Prepare matrices once
  //   const A = randomMatrix(size, size, density);
  //   const B = randomMatrix(size, size, density);
  //   const ADense = new Matrix(A);
  //   const BDense = new Matrix(B);

  //   // Benchmark the multiplication
  //   yield () => do_not_optimize(ADense.mmul(BDense));
  // }).range('size', min, max, multiplier);
});

await run();

import { run, bench, group, do_not_optimize } from 'mitata';
import { Matrix } from 'ml-matrix';
import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld2.js';
import { randomSparseMatrix } from './utils/randomSparseMatrix.js';

const sizes = [64, 128, 256];
const densities = [0.01, 0.015, 0.02, 0.025, 0.03];

for (const density of densities) {
  for (const size of sizes) {
    const A = randomSparseMatrix(size, size, density);
    const B = randomSparseMatrix(size, size, density);
    let denseA = A.to2DArray();
    let denseB = B.to2DArray();
    const AOld = new SparseMatrixOld(denseA);
    const BOld = new SparseMatrixOld(denseB);
    denseA = new Matrix(denseA);
    denseB = new Matrix(denseB);

    group(`size:${size}-density:${density}`, () => {
      bench('mmulNew', () => {
        do_not_optimize(A.mmul(B));
      }); //.gc('inner');
      bench('mmul', () => {
        do_not_optimize(AOld.mmul(BOld));
      }); //.gc('inner');
      bench('denseMatrix', () => {
        do_not_optimize(denseA.mmul(denseB));
      }); //.gc('inner');
    });
  }
}

await run({ silent: false });

import { run, bench, group, do_not_optimize } from 'mitata';
import { Matrix } from 'ml-matrix';
import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld2.js';
import fs from 'fs';
import { randomSparseMatrix } from './utils/randomSparseMatrix.js';

const sizes = [64, 128, 256];
const densities = [0.01, 0.015, 0.02, 0.025, 0.03];
const results = [];

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

    // Warm up
    A.mmul(B);

    let mmulNewAvg, mmulAvg, denseAvg;

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

    // mitata will handle timing and reporting
    // You can extract results from mitata's output or use the save() function
  }
}

await run({ silent: false });
// await save({ file: './benchmark/mitata_results.json', format: 'json' });

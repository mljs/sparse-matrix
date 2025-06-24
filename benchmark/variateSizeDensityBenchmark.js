/* eslint-disable no-invalid-this */
import fs from 'fs';

import Benchmark from 'benchmark';
import { Matrix } from 'ml-matrix';

import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld.js';
import { randomSparseMatrix } from './utils/randomSparseMatrix.js';

function runSizeSweepBenchmark() {
  const sizes = [32, 64, 128, 256];
  const densities = [0.01, 0.015, 0.02, 0.025, 0.03];
  const results = [];

  for (const densityA of densities) {
    for (const densityB of densities) {
      for (const m of sizes) {
        for (const n of sizes) {
          for (const p of sizes) {
            const A = randomSparseMatrix(m, n, densityA);
            const B = randomSparseMatrix(n, p, densityB);
            let denseA = A.to2DArray();
            let denseB = B.to2DArray();
            const AOld = new SparseMatrixOld(denseA);
            const BOld = new SparseMatrixOld(denseB);
            denseA = new Matrix(denseA);
            denseB = new Matrix(denseB);
            // Use Benchmark.js for each method
            const suite = new Benchmark.Suite();
            let mmulNewAvg, mmulAvg, denseAvg;
            suite
              .add('mmulNew', () => {
                A.mmul(B);
              })
              .add('mmul', () => {
                AOld.mmul(BOld);
              })
              .add('denseMatrix', () => {
                denseA.mmul(denseB);
              })
              .on('cycle', (event) => {
                // Optionally log: console.log(String(event.target));
              })
              .on('complete', function onComplete() {
                this.forEach((bench) => {
                  if (bench.name === 'mmulNew') mmulNewAvg = 1000 / bench.hz;
                  if (bench.name === 'mmul') mmulAvg = 1000 / bench.hz;
                  if (bench.name === 'denseMatrix') denseAvg = 1000 / bench.hz;
                });
              })
              .run({ async: false });
            results.push({
              densityA,
              densityB,
              aShape: [m, n],
              bShape: [n, p],
              dense: denseAvg,
              new: mmulNewAvg,
              old: mmulAvg,
            });
          }
        }
      }
    }
  }
  fs.writeFileSync(
    './benchmark/size_sweep_results-dense.json',
    JSON.stringify(results, null, 2),
  );
  console.log('Size sweep benchmark results saved to size_sweep_results.json');
}

runSizeSweepBenchmark();

import Benchmark from 'benchmark';
import { Matrix } from 'ml-matrix';
import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld.js';
import fs from 'fs';
import { randomSparseMatrix } from './utils/randomSparseMatrix.js';

function runSizeSweepBenchmarkCompact() {
  const sizes = [32, 64, 128, 256];
  const densities = [0.01, 0.015, 0.02, 0.025, 0.03];
  const results = [];

  for (const densityA of densities) {
    for (const m of sizes) {
      const A = randomSparseMatrix(m, m, densityA);
      const B = randomSparseMatrix(m, m, densityA);
      let denseA = A.to2DArray();
      let denseB = B.to2DArray();
      const AOld = new SparseMatrixOld(denseA);
      const BOld = new SparseMatrixOld(denseB);
      denseA = new Matrix(denseA);
      denseB = new Matrix(denseB);
      // Warm up
      A.mmul(B);
      // Use Benchmark.js for each method
      const suite = new Benchmark.Suite();
      let mmulNewAvg, mmulAvg, denseAvg;
      suite
        .add('mmulNew', function () {
          A.mmul(B);
        })
        .add('mmul', function () {
          AOld.mmul(BOld);
        })
        .add('denseMatrix', function () {
          denseA.mmul(denseB);
        })
        .on('cycle', function (event) {
          // Optionally log: console.log(String(event.target));
        })
        .on('complete', function () {
          this.forEach((bench) => {
            if (bench.name === 'mmulNew') mmulNewAvg = 1000 / bench.hz;
            if (bench.name === 'mmul') mmulAvg = 1000 / bench.hz;
            if (bench.name === 'denseMatrix') denseAvg = 1000 / bench.hz;
          });
        })
        .run({ async: false });
      results.push({
        density: densityA,
        size: m,
        dense: denseAvg,
        new: mmulNewAvg,
        old: mmulAvg,
      });
    }
  }
  fs.writeFileSync(
    './benchmark/size_sweep_results-dense.json',
    JSON.stringify(results, null, 2),
  );
  console.log('Size sweep benchmark results saved to size_sweep_results.json');
}

runSizeSweepBenchmarkCompact();

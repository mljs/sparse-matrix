import { SparseMatrix } from '../src/index.js';
import { Matrix } from 'ml-matrix';
import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld.js';
import fs from 'fs';

function randomSparseMatrix(rows, cols, density = 0.01) {
  const matrix = [];
  for (let i = 0; i < rows; i++) {
    const row = new Array(cols).fill(0);
    for (let j = 0; j < cols; j++) {
      if (Math.random() < density) {
        row[j] = Math.random() * 10;
      }
    }
    matrix.push(row);
  }
  return new SparseMatrix(matrix);
}

function benchmark(fn, label, iterations = 5) {
  const times = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    fn();
    const t1 = performance.now();
    times.push(t1 - t0);
  }
  const avg = times.reduce((a, b) => a + b, 0) / times.length;
  console.log(`${label}: avg ${avg.toFixed(2)} ms over ${iterations} runs`);
  return avg;
}

function printWinner(label1, avg1, label2, avg2) {
  let winner, loser, win, lose;
  if (avg1 < avg2) {
    winner = label1;
    win = avg1;
    loser = label2;
    lose = avg2;
  } else {
    winner = label2;
    win = avg2;
    loser = label1;
    lose = avg1;
  }

  const percent = ((lose - win) / lose) * 100;
  console.log(
    `  -> ${winner} was ${(lose / win).toFixed(2)}x faster (${percent.toFixed(
      1,
    )}% faster) than ${loser}\n`,
  );
}

function runBenchmarks() {
  const m = 128;
  const n = 128;
  const p = 128;
  const densityA = 0.01;
  const densityB = 0.01;

  const nbIterations = 3;
  const A = randomSparseMatrix(m, n, densityA);
  const B = randomSparseMatrix(n, p, densityB);

  let denseA = A.to2DArray();
  let denseB = B.to2DArray();

  const AOld = new SparseMatrixOld(denseA);
  const BOld = new SparseMatrixOld(denseB);

  denseA = new Matrix(denseA);
  denseB = new Matrix(denseB);

  denseA.mmul(denseB);
  // 1. add vs addNew
  // const addAvg = benchmark(() => {
  //   const a = AOld.clone();
  //   a.add(BOld);
  // }, 'add');

  // const addNewAvg = benchmark(() => {
  //   const a = A.clone();
  //   a.add(B);
  // }, 'addNew');

  // printWinner('add', addAvg, 'addNew', addNewAvg);

  // 2. mmul vs mmulNew

  const mmulNewAvg = benchmark(
    () => {
      A.mmul(B);
    },
    'mmulNew',
    nbIterations,
  );

  const mmulAvg = benchmark(
    () => {
      AOld.mmul(BOld);
    },
    'mmul',
    nbIterations,
  );

  const denseAvg = benchmark(
    () => {
      denseA.mmul(denseB);
    },
    'denseMatrix',
    nbIterations,
  );

  printWinner('mmul', mmulAvg, 'mmulNew', mmulNewAvg);

  // 3. kroneckerProduct vs kroneckerProductNew
  // const kronNewAvg = benchmark(() => {
  //   A.kroneckerProduct(B);
  // }, 'kroneckerProductNew');
  // const kronAvg = benchmark(() => {
  //   AOld.kroneckerProduct(BOld);
  // }, 'kroneckerProduct');

  // printWinner('kroneckerProduct', kronAvg, 'kroneckerProductNew', kronNewAvg);

  // 4. matrix multiplication
  // const mulAvg = benchmark(() => {
  //   A.mul(5);
  // }, 'mul');

  // const mulNewAvg = benchmark(() => {
  //   AOld.mul(5);
  // }, 'mulNew');

  // printWinner('mul', mulAvg, 'mulNew', mulNewAvg);
}

function runSizeSweepBenchmark() {
  const sizes = [8, 16, 32, 64, 128];
  const densities = [0.01, 0.015, 0.02, 0.025, 0.03, 0.35];
  const results = [];

  for (const densityA of densities) {
    for (const densityB of densities) {
      for (const m of sizes) {
        for (const n of sizes) {
          for (const p of sizes) {
            // A: m x n, B: n x p

            const A = randomSparseMatrix(m, n, densityA);
            const B = randomSparseMatrix(n, p, densityB);

            let denseA = A.to2DArray();
            let denseB = B.to2DArray();

            const AOld = new SparseMatrixOld(denseA);
            const BOld = new SparseMatrixOld(denseB);

            denseA = new Matrix(denseA);
            denseB = new Matrix(denseB);

            const mmulNewAvg = benchmark(
              () => {
                A.mmul(B);
              },
              'mmulNew',
              3,
            );

            const mmulAvg = benchmark(
              () => {
                AOld.mmul(BOld);
              },
              'mmul',
              3,
            );

            const denseAvg = benchmark(() => {
              denseA.mmul(denseB), 'denseMatrix', 3;
            });

            results.push({
              densityA,
              densityB,
              A_shape: [m, n],
              B_shape: [n, p],
              dense: denseAvg,
              mmulNew: mmulNewAvg,
              mmul: mmulAvg,
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

runBenchmarks();
// Uncomment to run the size sweep benchmark
// runSizeSweepBenchmark();

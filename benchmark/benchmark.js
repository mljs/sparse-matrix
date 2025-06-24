/* eslint-disable no-invalid-this */
import Benchmark from 'benchmark';
import { Matrix } from 'ml-matrix';

import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld.js';
import { printWinner } from './utils/printWinner.js';
import { randomSparseMatrix } from './utils/randomSparseMatrix.js';

function runBenchmarks() {
  const m = 256;
  const n = 256;
  const p = 256;
  const densityA = 0.01;
  const densityB = 0.01;

  const A = randomSparseMatrix(m, n, densityA);
  const B = randomSparseMatrix(n, p, densityB);

  let denseA = A.to2DArray();
  let denseB = B.to2DArray();

  const AOld = new SparseMatrixOld(denseA);
  const BOld = new SparseMatrixOld(denseB);

  denseA = new Matrix(denseA);
  denseB = new Matrix(denseB);

  const results = [];

  // 1. add vs addNew
  const addSuite = new Benchmark.Suite('add');
  addSuite
    .add('add', () => {
      const a = AOld.clone();
      a.add(BOld);
    })
    .add('addNew', () => {
      const a = A.clone();
      a.add(B);
    })
    .add('addDense', () => {
      const a = denseA.clone();
      a.add(denseB);
    })
    .on('cycle', (event) => {
      console.log(String(event.target));
    })
    .on('complete', function onComplete() {
      const fastest = this.filter('fastest').map('name');
      this.forEach((bench) => {
        results.push({ label: bench.name, avg: 1000 / bench.hz });
      });
      printWinner(results);
    })
    .run({ async: false });

  // 2. mmul vs mmulNew
  const mmulSuite = new Benchmark.Suite('mmul');
  mmulSuite
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
      console.log(String(event.target));
    })
    .on('complete', function onComplete() {
      const mmulResults = [];
      this.forEach((bench) => {
        mmulResults.push({ label: bench.name, avg: 1000 / bench.hz });
      });
      printWinner(mmulResults);
    })
    .run({ async: false });

  // 3. kroneckerProduct vs kroneckerProductNew
  const kronSuite = new Benchmark.Suite('kroneckerProduct');
  kronSuite
    .add('kroneckerProductNew', () => {
      A.kroneckerProduct(B);
    })
    .add('kroneckerProduct', () => {
      AOld.kroneckerProduct(BOld);
    })
    .add('kroneckerProductDense', () => {
      denseA.kroneckerProduct(denseB);
    })
    .on('cycle', (event) => {
      console.log(String(event.target));
    })
    .on('complete', function onComplete() {
      const kronResults = [];
      this.forEach((bench) => {
        kronResults.push({ label: bench.name, avg: 1000 / bench.hz });
      });
      printWinner(kronResults);
    })
    .run({ async: false });

  // 4. matrix multiplication
  const mulSuite = new Benchmark.Suite('mul');
  mulSuite
    .add('mul', () => {
      A.mul(5);
    })
    .add('mulNew', () => {
      AOld.mul(5);
    })
    .on('cycle', (event) => {
      console.log(String(event.target));
    })
    .on('complete', function onComplete() {
      const mulResults = [];
      this.forEach((bench) => {
        mulResults.push({ label: bench.name, avg: 1000 / bench.hz });
      });
      printWinner(mulResults);
    })
    .run({ async: false });
}

runBenchmarks();

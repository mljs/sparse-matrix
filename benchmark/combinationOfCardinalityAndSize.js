import { run, bench, do_not_optimize, lineplot } from 'mitata';
import { xSequentialFillFromStep } from 'ml-spectra-processing';

import { SparseMatrix } from '../src/index.js';

import { randomMatrix } from './utils/randomMatrix.js';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';

/* eslint 
func-names: 0
camelcase: 0
*/
// Prepare matrices once
const cardinalities = Array.from(
  xSequentialFillFromStep({ from: 10, step: 5, size: 2 }),
);

// const dimensions = Array.from(
//   xSequentialFillFromStep({ from: 700, step: 100, size: 13 }),
// );

const dimensions = [512];
lineplot(() => {
  bench('hibrid($cardinality,$dimension)', function* (ctx) {
    const cardinality = ctx.get('cardinality');
    const size = ctx.get('dimension');
    // Prepare matrices once
    let A = new SparseMatrix(randomMatrix(size, size, cardinality));
    let B = new SparseMatrix(randomMatrix(size, size, cardinality));

    // Benchmark the multiplication
    yield () => do_not_optimize(A.mmul(B));
    do_not_optimize(A);
    do_not_optimize(B);
  })
    .gc('inner')
    .args('cardinality', cardinalities) //.range('size', 32, 1024, 2); //.args('size', sizes);
    .args('dimension', dimensions);

  bench('small($cardinality,$dimension)', function* (ctx) {
    const cardinality = ctx.get('cardinality');
    const size = ctx.get('dimension');
    // Prepare matrices once
    let A = new SparseMatrix(randomMatrix(size, size, cardinality));
    let B = new SparseMatrix(randomMatrix(size, size, cardinality));

    // Benchmark the multiplication
    yield () => do_not_optimize(A._mmulSmall(B));
    // Explicit cleanup
    do_not_optimize(A);
    do_not_optimize(B);
  })
    .gc('inner')
    .args('cardinality', cardinalities) //.range('size', 32, 1024, 2); //.args('size', sizes);
    .args('dimension', dimensions);

  bench('low($cardinality,$dimension)', function* (ctx) {
    const cardinality = ctx.get('cardinality');
    const size = ctx.get('dimension');
    // Prepare matrices once
    let A = new SparseMatrix(randomMatrix(size, size, cardinality));
    let B = new SparseMatrix(randomMatrix(size, size, cardinality));

    // Benchmark the multiplication
    yield () => do_not_optimize(A._mmulLowDensity(B));
    // Explicit cleanup
    do_not_optimize(A);
    do_not_optimize(B);
  })
    .gc('inner')
    .args('cardinality', cardinalities) //.range('size', 32, 1024, 2); //.args('size', sizes);
    .args('dimension', dimensions);

  bench('medium($cardinality,$dimension)', function* (ctx) {
    const cardinality = ctx.get('cardinality');
    const size = ctx.get('dimension');
    // Prepare matrices once
    let A = new SparseMatrix(randomMatrix(size, size, cardinality));
    let B = new SparseMatrix(randomMatrix(size, size, cardinality));

    // Benchmark the multiplication
    yield () => {
      do_not_optimize(A._mmulMediumDensity(B));
    };

    // Explicit cleanup
    do_not_optimize(A);
    do_not_optimize(B);
  })
    .gc('inner')
    .args('cardinality', cardinalities) //.range('size', 32, 1024, 2); //.args('size', sizes);
    .args('dimension', dimensions);
});

// Run benchmarks and capture results
const results = await run({
  // Force GC between every benchmark
  gc: true,
  // // More samples for statistical significance
  // min_samples: 20,
  // max_samples: 200,
  // // Longer warmup to stabilize CPU state
  // warmup_samples: 10,
  // warmup_threshold: 100, // ms
  // Longer minimum time for stable measurements
  // min_cpu_time: 2000, // 2 seconds minimum
  // // Batch settings to reduce variance
  // batch_samples: 5,
  // batch_threshold: 10, // ms
  // // Enable colors
  // colors: true,
});

// Process and store results
const processedResults = [];

for (const benchmark of results.benchmarks) {
  for (const run of benchmark.runs) {
    if (run.stats) {
      processedResults.push({
        name: benchmark.alias,
        cardinality: run.args.cardinality,
        dimension: run.args.dimension,
        avg: run.stats.avg,
        min: run.stats.min,
        max: run.stats.max,
        p50: run.stats.p50,
        p75: run.stats.p75,
        p99: run.stats.p99,
        samples: run.stats.samples.length,
        ticks: run.stats.ticks,
      });
    }
  }
}

// Save results to JSON file
await writeFile(
  path.join(import.meta.dirname, `benchmark-results.json`),
  JSON.stringify(processedResults, null, 2),
);

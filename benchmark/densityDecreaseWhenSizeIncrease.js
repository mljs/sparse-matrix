import { run, bench, group, do_not_optimize } from 'mitata';
// import { Matrix } from 'ml-matrix';

import { SparseMatrix as SparseMatrixOld } from './class/SparseMatrixOld.js';
import { randomMatrix } from './utils/randomMatrix.js';
import { SparseMatrix } from '../src/index.js';

/* eslint 
func-names: 0 
camelcase: 0
*/

const sizes = [8, 16, 32, 256, 512, 1024];
const densities = [0.125, 0.0625, 0.03125, 0.0039, 0.00197, 0.001];

for (let i = 0; i < sizes.length; i++) {
  const size = sizes[i];
  const density = densities[i];
  const denseA = randomMatrix(size, size, density);
  const denseB = randomMatrix(size, size, density);
  const AOld = new SparseMatrixOld(denseA);
  const BOld = new SparseMatrixOld(denseB);

  const A = new SparseMatrix(denseA);
  const B = new SparseMatrix(denseB);

  group(`size:${size}-density:${density}`, () => {
    bench('mmulNew', () => {
      do_not_optimize(A.mmul(B));
    }).gc('inner');
    bench('mmul', () => {
      do_not_optimize(AOld.mmul(BOld));
    }).gc('inner');
  });
}

await run({ silent: false });

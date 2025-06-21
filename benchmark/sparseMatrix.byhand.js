import { SparseMatrix } from '../lib/index.js';

let size = 200;

console.time('dense');

const matrix1 = new SparseMatrix(size, size);
fillSparseMatrix(matrix1);

const matrix2 = new SparseMatrix(size, size);
fillSparseMatrix(matrix2);

const product = new SparseMatrix(size, size);
for (let i = 0; i < size; i++) {
  for (let j = 0; j < size; j++) {
    for (let k = 0; k < size; k++) {
      product.set(
        i,
        j,
        product.get(i, j) + matrix1.get(i, k) * matrix2.get(k, j),
      );
    }
  }
}

let sum = 0;
for (let i = 0; i < size; i++) {
  for (let j = 0; j < size; j++) {
    sum += product.get(i, j);
  }
}
console.timeEnd('dense');
console.log(sum);

function fillSparseMatrix(matrix) {
  for (let row = 0; row < matrix.rows; row++) {
    for (let column = 0; column < matrix.columns; column++) {
      matrix.set(row, column, 1);
    }
  }
}

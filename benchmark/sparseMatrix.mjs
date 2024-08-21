import { SparseMatrix } from '../lib/index.js';

let size = 200;

console.time('dense');

const matrix = new SparseMatrix(size, size);
fillSparseMatrix(matrix)

const matrix2 = new SparseMatrix(size, size);
fillSparseMatrix(matrix2)

const product = matrix.mmul(matrix2);
// sum of all the elements
let sum = 0;
for (let i = 0; i < size; i++) {
  for (let j = 0; j < size; j++) {
    sum += product.get(i, j);
  }
}
console.timeEnd('dense');
console.log(sum)

function fillSparseMatrix(matrix) {
  for (let row = 0; row < matrix.rows; row++) {
    for (let column = 0; column < matrix.columns; column++) {
      matrix.set(row, column, 1);
    }
  }
}
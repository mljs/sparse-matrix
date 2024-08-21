import { Matrix } from 'ml-matrix';

let size = 200;

console.time('dense');
const matrix = new Matrix(size, size).fill(1);
const matrix2 = new Matrix(size, size).fill(1);

const product = matrix.mmul(matrix2);
// sum of all the elements
let sum = 0;
for (let i = 0; i < size; i++) {
  for (let j = 0; j < size; j++) {
    sum += product.get(i, j);
  }
}
console.log(sum)
console.timeEnd('dense');


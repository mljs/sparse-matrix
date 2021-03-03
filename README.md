# sparse-matrix

[![NPM version][npm-image]][npm-url]
[![coverage status][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

Sparse matrix library.

## Installation

`$ npm i ml-sparse-matrix`

## Usage

```js
import { SparseMatrix } from "ml-sparse-matrix";

const matrix1 = new SparseMatrix([
  [1, 2],
  [3, 4],
]);
const matrix2 = new SparseMatrix([
  [0, 5],
  [6, 7],
]);
const product = matrix1.kroneckerProduct(matrix2);
```

## [API Documentation](https://mljs.github.io/sparse-matrix/)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-sparse-matrix.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-sparse-matrix
[codecov-image]: https://codecov.io/github/mljs/sparse-matrix/coverage.svg?style=flat-square
[codecov-url]: https://codecov.io/github/mljs/sparse-matrix
[download-image]: https://img.shields.io/npm/dm/ml-sparse-matrix.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-sparse-matrix

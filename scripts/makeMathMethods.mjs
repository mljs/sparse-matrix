/* eslint-disable no-console */

const inplaceOperator = `
  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  %name%(value) {
    if (typeof value === 'number') return this.%name%S(value);
    return this.%name%M(value);
  }`;

const inplaceOperatorScalar = `
  /**
   * @param {number} value
   * @returns {this}
   */
  %name%(value) {
    this.forEachNonZero((i, j, v) => v %op% value);
    return this;
  }`;

const inplaceOperatorMatrix = `
  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  %name%(matrix) {
    matrix.forEachNonZero((i, j, v) => {
        this.set(i, j, this.get(i, j) %op% v);
        return v;
    });
    return this;
  }`;

const staticOperator = `
  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static %name%(matrix, value) {
    return new SparseMatrix(matrix).%name%S(value);
  }`;

const inplaceMethod = `
  /**
   * @returns {this}
   */
  %name%() {
    this.forEachNonZero((i, j, v) => %method%(v));
    return this;
  }`;

const staticMethod = `
  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static %name%(matrix) {
    return new SparseMatrix(matrix).%name%();
  }`;

const operators = [
  // Arithmetic operators
  ['+', 'add'],
  ['-', 'sub', 'subtract'],
  ['*', 'mul', 'multiply'],
  ['/', 'div', 'divide'],
  ['%', 'mod', 'modulus'],
  // Bitwise operators
  ['&', 'and'],
  ['|', 'or'],
  ['^', 'xor'],
  ['<<', 'leftShift'],
  ['>>', 'signPropagatingRightShift'],
  ['>>>', 'rightShift', 'zeroFillRightShift'],
];

for (const operator of operators) {
  for (let i = 1; i < operator.length; i++) {
    console.log(
      fillTemplateFunction(inplaceOperator, {
        name: operator[i],
        op: operator[0],
      }),
    );
    console.log(
      fillTemplateFunction(inplaceOperatorScalar, {
        name: `${operator[i]}S`,
        op: operator[0],
      }),
    );
    console.log(
      fillTemplateFunction(inplaceOperatorMatrix, {
        name: `${operator[i]}M`,
        op: operator[0],
      }),
    );

    console.log(fillTemplateFunction(staticOperator, { name: operator[i] }));
  }
}

let methods = [['~', 'not']];

for (const mathMethod of [
  'abs',
  'acos',
  'acosh',
  'asin',
  'asinh',
  'atan',
  'atanh',
  'cbrt',
  'ceil',
  'clz32',
  'cos',
  'cosh',
  'exp',
  'expm1',
  'floor',
  'fround',
  'log',
  'log1p',
  'log10',
  'log2',
  'round',
  'sign',
  'sin',
  'sinh',
  'sqrt',
  'tan',
  'tanh',
  'trunc',
]) {
  methods.push([`Math.${mathMethod}`, mathMethod]);
}

for (const method of methods) {
  for (let i = 1; i < method.length; i++) {
    console.log(
      fillTemplateFunction(inplaceMethod, {
        name: method[i],
        method: method[0],
      }),
    );
    console.log(fillTemplateFunction(staticMethod, { name: method[i] }));
  }
}

function fillTemplateFunction(template, values) {
  for (const i in values) {
    template = template.replaceAll(new RegExp(`%${i}%`, 'g'), values[i]);
  }
  return template;
}

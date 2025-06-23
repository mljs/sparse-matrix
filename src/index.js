import HashTable from 'ml-hash-table';

/** @typedef {(row: number, column: number, value: number) => number | false} ForEachNonZeroCallback */

export class SparseMatrix {
  constructor(rows, columns, options = {}) {
    if (rows instanceof SparseMatrix) {
      // clone
      const other = rows;
      this._init(
        other.rows,
        other.columns,
        other.elements.clone(),
        other.threshold,
      );
      return;
    }

    if (Array.isArray(rows)) {
      const matrix = rows;
      const nbRows = matrix.length;
      const nbColmuns = matrix[0].length;
      options = columns || { initialCapacity: nbRows * nbColmuns };
      columns = matrix[0].length;
      this._init(nbRows, columns, new HashTable(options), options.threshold);
      for (let i = 0; i < nbRows; i++) {
        for (let j = 0; j < columns; j++) {
          let value = matrix[i][j];
          if (this.threshold && Math.abs(value) < this.threshold) value = 0;
          if (value !== 0) {
            this.elements.set(i * columns + j, matrix[i][j]);
          }
        }
      }
      this.elements.maybeShrinkCapacity();
    } else {
      this._init(rows, columns, new HashTable(options), options.threshold);
    }
  }

  _init(rows, columns, elements, threshold) {
    this.rows = rows;
    this.columns = columns;
    this.elements = elements;
    this.threshold = threshold || 0;
  }

  static eye(rows = 1, columns = rows) {
    const min = Math.min(rows, columns);
    const matrix = new SparseMatrix(rows, columns, { initialCapacity: min });
    for (let i = 0; i < min; i++) {
      matrix.set(i, i, 1);
    }
    return matrix;
  }

  clone() {
    return new SparseMatrix(this);
  }

  /**
   * @returns {number[][]}
   */
  to2DArray() {
    const copy = new Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      copy[i] = new Array(this.columns);
      for (let j = 0; j < this.columns; j++) {
        copy[i][j] = this.get(i, j);
      }
    }
    return copy;
  }

  isSquare() {
    return this.rows === this.columns;
  }

  isSymmetric() {
    if (!this.isSquare()) return false;

    let symmetric = true;
    this.forEachNonZero((i, j, v) => {
      if (this.get(j, i) !== v) {
        symmetric = false;
        return false;
      }
      return v;
    });
    return symmetric;
  }

  /**
   * Search for the wither band in the main diagonals
   * @returns {number}
   */
  bandWidth() {
    let min = this.columns;
    let max = -1;
    this.forEachNonZero((i, j, v) => {
      let diff = i - j;
      min = Math.min(min, diff);
      max = Math.max(max, diff);
      return v;
    });
    return max - min;
  }

  /**
   * Test if a matrix is consider banded using a threshold
   * @param {number} width
   * @returns {boolean}
   */
  isBanded(width) {
    let bandWidth = this.bandWidth();
    return bandWidth <= width;
  }

  /**
   * @returns {number}
   */
  get cardinality() {
    return this.elements.size;
  }

  /**
   * @returns {number}
   */
  get size() {
    return this.rows * this.columns;
  }

  /**
   * @param {number} row
   * @param {number} column
   * @returns {number}
   */
  get(row, column) {
    return this.elements.get(row * this.columns + column);
  }

  /**
   * @param {number} row
   * @param {number} column
   * @param {number} value
   * @returns {this}
   */
  set(row, column, value) {
    if (this.threshold && Math.abs(value) < this.threshold) value = 0;
    if (value === 0) {
      this.elements.remove(row * this.columns + column);
    } else {
      this.elements.set(row * this.columns + column, value);
    }

    return this;
  }

  mmul(other) {
    if (this.columns !== other.rows) {
      // eslint-disable-next-line no-console
      console.warn(
        'Number of columns of left matrix are not equal to number of rows of right matrix.',
      );
    }

    const m = this.rows;
    const p = other.columns;

    const result = matrixCreateEmpty(m, p);

    const {
      columns: otherCols,
      rows: otherRows,
      values: otherValues,
    } = other.getNonZeros({ format: 'csr' });
    const {
      columns: thisCols,
      rows: thisRows,
      values: thisValues,
    } = this.getNonZeros({ format: 'csc' });

    const thisNbCols = this.columns;
    for (let t = 0; t < thisNbCols; t++) {
      const tStart = thisCols[t];
      const tEnd = thisCols[t + 1];
      const oStart = otherRows[t];
      const oEnd = otherRows[t + 1];
      for (let f = tStart; f < tEnd; f++) {
        for (let k = oStart; k < oEnd; k++) {
          const i = thisRows[f];
          const l = otherCols[k];
          result[i][l] += thisValues[f] * otherValues[k];
        }
      }
    }

    return new SparseMatrix(result);
  }

  /**
   * @param {SparseMatrix} other
   * @returns {SparseMatrix}
   */
  kroneckerProduct(other) {
    const m = this.rows;
    const n = this.columns;
    const p = other.rows;
    const q = other.columns;

    const result = new SparseMatrix(m * p, n * q, {
      initialCapacity: this.cardinality * other.cardinality + 20,
    });

    const {
      columns: otherCols,
      rows: otherRows,
      values: otherValues,
    } = other.getNonZeros();
    const {
      columns: thisCols,
      rows: thisRows,
      values: thisValues,
    } = this.getNonZeros();

    const nbThisActive = thisCols.length;
    const nbOtherActive = otherCols.length;
    for (let t = 0; t < nbThisActive; t++) {
      const pi = p * thisRows[t];
      const qj = q * thisCols[t];
      for (let o = 0; o < nbOtherActive; o++) {
        result.set(
          pi + otherRows[o],
          qj + otherCols[o],
          otherValues[o] * thisValues[t],
        );
      }
    }

    return result;
  }

  withEachNonZero(callback, sort = false) {
    const { state, table, values } = this.elements;
    const nbStates = state.length;
    const activeIndex = new Float64Array(this.cardinality);
    for (let i = 0, j = 0; i < nbStates; i++) {
      if (state[i] === 1) activeIndex[j++] = i;
    }

    if (sort) {
      activeIndex.sort((a, b) => table[a] - table[b]);
    }

    const columns = this.columns;
    for (const i of activeIndex) {
      const key = table[i];
      callback((key / columns) | 0, key % columns, values[i]);
    }
  }

  /**
   * Calls `callback` for each value in the matrix that is not zero.
   * The callback can return:
   * - `false` to stop the iteration immediately
   * - the same value to keep it unchanged
   * - 0 to remove the value
   * - something else to change the value
   * @param {ForEachNonZeroCallback} callback
   * @returns {this}
   */
  forEachNonZero(callback) {
    this.elements.forEachPair((key, value) => {
      const i = (key / this.columns) | 0;
      const j = key % this.columns;
      let r = callback(i, j, value);
      if (r === false) return false; // stop iteration
      if (this.threshold && Math.abs(r) < this.threshold) r = 0;
      if (r !== value) {
        if (r === 0) {
          this.elements.remove(key, true);
        } else {
          this.elements.set(key, r);
        }
      }
      return true;
    });
    this.elements.maybeShrinkCapacity();
    return this;
  }

  getNonZeros(options = {}) {
    const cardinality = this.cardinality;
    /** @type {Float64Array} */
    const rows = new Float64Array(cardinality);
    /** @type {Float64Array} */
    const columns = new Float64Array(cardinality);
    /** @type {Float64Array} */
    const values = new Float64Array(cardinality);

    const { format, sort = false } = options;

    let idx = 0;
    this.withEachNonZero((i, j, value) => {
      rows[idx] = i;
      columns[idx] = j;
      values[idx] = value;
      idx++;
    }, sort || format);

    if (!format) return { rows, columns, values };

    if (!['csr', 'csc'].includes(format.toLowerCase())) {
      throw new Error(`format ${format} is not supported`);
    }

    const csrMatrix = cooToCsr({ rows, columns, values }, this.rows);
    return format.toLowerCase() === 'csc'
      ? csrToCsc(csrMatrix, this.columns)
      : csrMatrix;
  }

  /**
   * @param {number} newThreshold
   * @returns {this}
   */
  setThreshold(newThreshold) {
    if (newThreshold !== 0 && newThreshold !== this.threshold) {
      this.threshold = newThreshold;
      this.forEachNonZero((i, j, v) => v);
    }
    return this;
  }

  /**
   * @returns {SparseMatrix} - New transposed sparse matrix
   */
  transpose() {
    let trans = new SparseMatrix(this.columns, this.rows, {
      initialCapacity: this.cardinality,
    });
    this.withEachNonZero((i, j, value) => {
      trans.set(j, i, value);
    });
    return trans;
  }

  /**
   * @returns {boolean}
   */
  isEmpty() {
    return this.rows === 0 || this.columns === 0;
  }

  // The following was generated by `makeMathMethods.mjs`
  //#region Math methods.
  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  add(value) {
    if (typeof value === 'number') return this.addS(value);
    return this.addM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  addS(value) {
    this.forEachNonZero((i, j, v) => v + value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  addM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) + v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static add(matrix, value) {
    return new SparseMatrix(matrix).addS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  sub(value) {
    if (typeof value === 'number') return this.subS(value);
    return this.subM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  subS(value) {
    this.forEachNonZero((i, j, v) => v - value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  subM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) - v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static sub(matrix, value) {
    return new SparseMatrix(matrix).subS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  subtract(value) {
    if (typeof value === 'number') return this.subtractS(value);
    return this.subtractM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  subtractS(value) {
    this.forEachNonZero((i, j, v) => v - value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  subtractM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) - v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static subtract(matrix, value) {
    return new SparseMatrix(matrix).subtractS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  mul(value) {
    if (typeof value === 'number') return this.mulS(value);
    return this.mulM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  mulS(value) {
    this.forEachNonZero((i, j, v) => v * value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  mulM(matrix) {
    if (typeof matrix !== 'number') {
      throw new RangeError('the argument should be a number');
    }

    if (matrix === 0) {
      this.elements = new HashTable();
    }

    this.withEachNonZero((i, j, v) => {
      this.set(i, j, v * matrix);
    });

    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static mul(matrix, value) {
    return new SparseMatrix(matrix).mulS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  multiply(value) {
    if (typeof value === 'number') return this.multiplyS(value);
    return this.multiplyM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  multiplyS(value) {
    this.forEachNonZero((i, j, v) => v * value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  multiplyM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) * v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static multiply(matrix, value) {
    return new SparseMatrix(matrix).multiplyS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  div(value) {
    if (typeof value === 'number') return this.divS(value);
    return this.divM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  divS(value) {
    this.forEachNonZero((i, j, v) => v / value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  divM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) / v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static div(matrix, value) {
    return new SparseMatrix(matrix).divS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  divide(value) {
    if (typeof value === 'number') return this.divideS(value);
    return this.divideM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  divideS(value) {
    this.forEachNonZero((i, j, v) => v / value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  divideM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) / v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static divide(matrix, value) {
    return new SparseMatrix(matrix).divideS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  mod(value) {
    if (typeof value === 'number') return this.modS(value);
    return this.modM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  modS(value) {
    this.forEachNonZero((i, j, v) => v % value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  modM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) % v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static mod(matrix, value) {
    return new SparseMatrix(matrix).modS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  modulus(value) {
    if (typeof value === 'number') return this.modulusS(value);
    return this.modulusM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  modulusS(value) {
    this.forEachNonZero((i, j, v) => v % value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  modulusM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) % v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static modulus(matrix, value) {
    return new SparseMatrix(matrix).modulusS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  and(value) {
    if (typeof value === 'number') return this.andS(value);
    return this.andM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  andS(value) {
    this.forEachNonZero((i, j, v) => v & value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  andM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) & v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static and(matrix, value) {
    return new SparseMatrix(matrix).andS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  or(value) {
    if (typeof value === 'number') return this.orS(value);
    return this.orM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  orS(value) {
    this.forEachNonZero((i, j, v) => v | value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  orM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) | v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static or(matrix, value) {
    return new SparseMatrix(matrix).orS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  xor(value) {
    if (typeof value === 'number') return this.xorS(value);
    return this.xorM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  xorS(value) {
    this.forEachNonZero((i, j, v) => v ^ value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  xorM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) ^ v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static xor(matrix, value) {
    return new SparseMatrix(matrix).xorS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  leftShift(value) {
    if (typeof value === 'number') return this.leftShiftS(value);
    return this.leftShiftM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  leftShiftS(value) {
    this.forEachNonZero((i, j, v) => v << value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  leftShiftM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) << v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static leftShift(matrix, value) {
    return new SparseMatrix(matrix).leftShiftS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  signPropagatingRightShift(value) {
    if (typeof value === 'number') {
      return this.signPropagatingRightShiftS(value);
    }
    return this.signPropagatingRightShiftM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  signPropagatingRightShiftS(value) {
    this.forEachNonZero((i, j, v) => v >> value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  signPropagatingRightShiftM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) >> v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static signPropagatingRightShift(matrix, value) {
    return new SparseMatrix(matrix).signPropagatingRightShiftS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  rightShift(value) {
    if (typeof value === 'number') return this.rightShiftS(value);
    return this.rightShiftM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  rightShiftS(value) {
    this.forEachNonZero((i, j, v) => v >>> value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  rightShiftM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) >>> v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static rightShift(matrix, value) {
    return new SparseMatrix(matrix).rightShiftS(value);
  }

  /**
   * @param {number|SparseMatrix} value
   * @returns {this}
   */
  zeroFillRightShift(value) {
    if (typeof value === 'number') return this.zeroFillRightShiftS(value);
    return this.zeroFillRightShiftM(value);
  }

  /**
   * @param {number} value
   * @returns {this}
   */
  zeroFillRightShiftS(value) {
    this.forEachNonZero((i, j, v) => v >>> value);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {this}
   */
  zeroFillRightShiftM(matrix) {
    matrix.forEachNonZero((i, j, v) => {
      this.set(i, j, this.get(i, j) >>> v);
      return v;
    });
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @param {number} value
   * @returns {SparseMatrix}
   */
  static zeroFillRightShift(matrix, value) {
    return new SparseMatrix(matrix).zeroFillRightShiftS(value);
  }

  /**
   * @returns {this}
   */
  not() {
    this.forEachNonZero((i, j, v) => ~v);
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static not(matrix) {
    return new SparseMatrix(matrix).not();
  }

  /**
   * @returns {this}
   */
  abs() {
    this.forEachNonZero((i, j, v) => Math.abs(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static abs(matrix) {
    return new SparseMatrix(matrix).abs();
  }

  /**
   * @returns {this}
   */
  acos() {
    this.forEachNonZero((i, j, v) => Math.acos(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static acos(matrix) {
    return new SparseMatrix(matrix).acos();
  }

  /**
   * @returns {this}
   */
  acosh() {
    this.forEachNonZero((i, j, v) => Math.acosh(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static acosh(matrix) {
    return new SparseMatrix(matrix).acosh();
  }

  /**
   * @returns {this}
   */
  asin() {
    this.forEachNonZero((i, j, v) => Math.asin(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static asin(matrix) {
    return new SparseMatrix(matrix).asin();
  }

  /**
   * @returns {this}
   */
  asinh() {
    this.forEachNonZero((i, j, v) => Math.asinh(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static asinh(matrix) {
    return new SparseMatrix(matrix).asinh();
  }

  /**
   * @returns {this}
   */
  atan() {
    this.forEachNonZero((i, j, v) => Math.atan(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static atan(matrix) {
    return new SparseMatrix(matrix).atan();
  }

  /**
   * @returns {this}
   */
  atanh() {
    this.forEachNonZero((i, j, v) => Math.atanh(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static atanh(matrix) {
    return new SparseMatrix(matrix).atanh();
  }

  /**
   * @returns {this}
   */
  cbrt() {
    this.forEachNonZero((i, j, v) => Math.cbrt(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static cbrt(matrix) {
    return new SparseMatrix(matrix).cbrt();
  }

  /**
   * @returns {this}
   */
  ceil() {
    this.forEachNonZero((i, j, v) => Math.ceil(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static ceil(matrix) {
    return new SparseMatrix(matrix).ceil();
  }

  /**
   * @returns {this}
   */
  clz32() {
    this.forEachNonZero((i, j, v) => Math.clz32(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static clz32(matrix) {
    return new SparseMatrix(matrix).clz32();
  }

  /**
   * @returns {this}
   */
  cos() {
    this.forEachNonZero((i, j, v) => Math.cos(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static cos(matrix) {
    return new SparseMatrix(matrix).cos();
  }

  /**
   * @returns {this}
   */
  cosh() {
    this.forEachNonZero((i, j, v) => Math.cosh(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static cosh(matrix) {
    return new SparseMatrix(matrix).cosh();
  }

  /**
   * @returns {this}
   */
  exp() {
    this.forEachNonZero((i, j, v) => Math.exp(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static exp(matrix) {
    return new SparseMatrix(matrix).exp();
  }

  /**
   * @returns {this}
   */
  expm1() {
    this.forEachNonZero((i, j, v) => Math.expm1(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static expm1(matrix) {
    return new SparseMatrix(matrix).expm1();
  }

  /**
   * @returns {this}
   */
  floor() {
    this.forEachNonZero((i, j, v) => Math.floor(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static floor(matrix) {
    return new SparseMatrix(matrix).floor();
  }

  /**
   * @returns {this}
   */
  fround() {
    this.forEachNonZero((i, j, v) => Math.fround(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static fround(matrix) {
    return new SparseMatrix(matrix).fround();
  }

  /**
   * @returns {this}
   */
  log() {
    this.forEachNonZero((i, j, v) => Math.log(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static log(matrix) {
    return new SparseMatrix(matrix).log();
  }

  /**
   * @returns {this}
   */
  log1p() {
    this.forEachNonZero((i, j, v) => Math.log1p(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static log1p(matrix) {
    return new SparseMatrix(matrix).log1p();
  }

  /**
   * @returns {this}
   */
  log10() {
    this.forEachNonZero((i, j, v) => Math.log10(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static log10(matrix) {
    return new SparseMatrix(matrix).log10();
  }

  /**
   * @returns {this}
   */
  log2() {
    this.forEachNonZero((i, j, v) => Math.log2(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static log2(matrix) {
    return new SparseMatrix(matrix).log2();
  }

  /**
   * @returns {this}
   */
  round() {
    this.forEachNonZero((i, j, v) => Math.round(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static round(matrix) {
    return new SparseMatrix(matrix).round();
  }

  /**
   * @returns {this}
   */
  sign() {
    this.forEachNonZero((i, j, v) => Math.sign(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static sign(matrix) {
    return new SparseMatrix(matrix).sign();
  }

  /**
   * @returns {this}
   */
  sin() {
    this.forEachNonZero((i, j, v) => Math.sin(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static sin(matrix) {
    return new SparseMatrix(matrix).sin();
  }

  /**
   * @returns {this}
   */
  sinh() {
    this.forEachNonZero((i, j, v) => Math.sinh(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static sinh(matrix) {
    return new SparseMatrix(matrix).sinh();
  }

  /**
   * @returns {this}
   */
  sqrt() {
    this.forEachNonZero((i, j, v) => Math.sqrt(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static sqrt(matrix) {
    return new SparseMatrix(matrix).sqrt();
  }

  /**
   * @returns {this}
   */
  tan() {
    this.forEachNonZero((i, j, v) => Math.tan(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static tan(matrix) {
    return new SparseMatrix(matrix).tan();
  }

  /**
   * @returns {this}
   */
  tanh() {
    this.forEachNonZero((i, j, v) => Math.tanh(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static tanh(matrix) {
    return new SparseMatrix(matrix).tanh();
  }

  /**
   * @returns {this}
   */
  trunc() {
    this.forEachNonZero((i, j, v) => Math.trunc(v));
    return this;
  }

  /**
   * @param {SparseMatrix} matrix
   * @returns {SparseMatrix}
   */
  static trunc(matrix) {
    return new SparseMatrix(matrix).trunc();
  }
  //#endregion
}

SparseMatrix.prototype.klass = 'Matrix';

SparseMatrix.identity = SparseMatrix.eye;
SparseMatrix.prototype.tensorProduct = SparseMatrix.prototype.kroneckerProduct;

function csrToCsc(csrMatrix, numCols) {
  const {
    values: csrValues,
    columns: csrColIndices,
    rows: csrRowPtr,
  } = csrMatrix;

  const cscValues = new Float64Array(csrValues.length);
  const cscRowIndices = new Float64Array(csrValues.length);
  const cscColPtr = new Float64Array(numCols + 1);

  for (let i = 0; i < csrColIndices.length; i++) {
    cscColPtr[csrColIndices[i] + 1]++;
  }

  for (let i = 1; i <= numCols; i++) {
    cscColPtr[i] += cscColPtr[i - 1];
  }

  const next = cscColPtr.slice();

  for (let row = 0; row < csrRowPtr.length - 1; row++) {
    for (let j = csrRowPtr[row], i = 0; j < csrRowPtr[row + 1]; j++, i++) {
      const col = csrColIndices[j];
      const pos = next[col];
      cscValues[pos] = csrValues[j];
      cscRowIndices[pos] = row;
      next[col]++;
    }
  }

  return { rows: cscRowIndices, columns: cscColPtr, values: cscValues };
}

function cooToCsr(cooMatrix, nbRows = 9) {
  const { values, columns, rows } = cooMatrix;
  const csrRowPtr = new Float64Array(nbRows + 1);
  const length = values.length;
  let currentRow = rows[0];
  for (let index = 0; index < length; ) {
    const prev = index;
    while (currentRow === rows[index] && index < length) ++index;
    csrRowPtr[currentRow + 1] = index;
    currentRow += 1;
  }
  return { rows: csrRowPtr, columns, values };
}

function matrixCreateEmpty(nbRows, nbColumns) {
  const newMatrix = [];
  for (let row = 0; row < nbRows; row++) {
    newMatrix.push(new Float64Array(nbColumns));
  }
  return newMatrix;
}

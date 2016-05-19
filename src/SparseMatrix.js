const HashTable = require('ml-hash-table');

class SparseMatrix {
    constructor(rows, columns, options = {}) {
        if (rows instanceof SparseMatrix) { // clone
            const other = rows;
            this._init(other.rows, other.columns, other.elements.clone(), other.threshold);
            return;
        }

        if (Array.isArray(rows)) {
            const matrix = rows;
            rows = matrix.length;
            options = columns || {};
            columns = matrix[0].length;
            this._init(rows, columns, new HashTable(options), options.threshold);
            for (var i = 0; i < rows; i++) {
                for (var j = 0; j < columns; j++) {
                    var value = matrix[i][j];
                    if (this.threshold && Math.abs(value) < this.threshold) value = 0;
                    if (value !== 0) {
                        this.elements.set(i * columns + j, matrix[i][j]);
                    }
                }
            }
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
        const matrix = new SparseMatrix(rows, columns, {initialCapacity: min});
        for (var i = 0; i < min; i++) {
            matrix.set(i, i, 1);
        }
        return matrix;
    }

    clone() {
        return new SparseMatrix(this);
    }
    
    to2DArray() {
        const copy = new Array(this.rows);
        for (var i = 0; i < this.rows; i++) {
            copy[i] = new Array(this.columns);
            for (var j = 0; j < this.columns; j++) {
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

        var symmetric = true;
        this.forEachNonZero((i, j, v) => {
            if (this.get(j, i) !== v) {
                symmetric = false;
                return false;
            }
            return v;
        });
        return symmetric;
    }

    get cardinality() {
        return this.elements.size;
    }

    get size() {
        return this.rows * this.columns;
    }

    get(row, column) {
        return this.elements.get(row * this.columns + column);
    }

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
        if (this.columns !== other.rows)
            console.warn('Number of columns of left matrix are not equal to number of rows of right matrix.');
        
        const m = this.rows;
        const p = other.columns;
        
        const result = new SparseMatrix(m, p);
        this.forEachNonZero((i, j, v1) => {
            other.forEachNonZero((k, l, v2) => {
                if (j === k) {
                    result.set(i, l, result.get(i, l) + v1 * v2);
                }
                return v2;
            });
            return v1;
        });
        return result;
    }

    kroneckerProduct(other) {
        const m = this.rows;
        const n = this.columns;
        const p = other.rows;
        const q = other.columns;

        const result = new SparseMatrix(m * p, n * q, {
            initialCapacity: this.cardinality * other.cardinality
        });
        this.forEachNonZero((i, j, v1) => {
            other.forEachNonZero((k, l, v2) => {
                result.set(p * i + k, q * j + l, v1 * v2);
                return v2;
            });
            return v1;
        });
        return result;
    }

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

    getNonZeros() {
        const cardinality = this.cardinality;
        const rows = new Array(cardinality);
        const columns = new Array(cardinality);
        const values = new Array(cardinality);
        var idx = 0;
        this.forEachNonZero((i, j, value) => {
            rows[idx] = i;
            columns[idx] = j;
            values[idx] = value;
            idx++;
            return value;
        });
        return {rows, columns, values};
    }

    setThreshold(newThreshold) {
        if (newThreshold !== 0 && newThreshold !== this.threshold) {
            this.threshold = newThreshold;
            this.forEachNonZero((i, j, v) => v);
        }
        return this;
    }
}

SparseMatrix.prototype.klass = 'Matrix';

SparseMatrix.identity = SparseMatrix.eye;
SparseMatrix.prototype.tensorProduct = SparseMatrix.prototype.kroneckerProduct;

module.exports = SparseMatrix;

/*
 Add dynamically instance and static methods for mathematical operations
 */

var inplaceOperator = `
(function %name%(value) {
    if (typeof value === 'number') return this.%name%S(value);
    return this.%name%M(value);
})
`;

var inplaceOperatorScalar = `
(function %name%S(value) {
    this.forEachNonZero((i, j, v) => v %op% value);
    return this;
})
`;

var inplaceOperatorMatrix = `
(function %name%M(matrix) {
    matrix.forEachNonZero((i, j, v) => {
        this.set(i, j, this.get(i, j) %op% v);
        return v;
    });
    return this;
})
`;

var staticOperator = `
(function %name%(matrix, value) {
    var newMatrix = new SparseMatrix(matrix);
    return newMatrix.%name%(value);
})
`;

var inplaceMethod = `
(function %name%() {
    this.forEachNonZero((i, j, v) => %method%(v));
    return this;
})
`;

var staticMethod = `
(function %name%(matrix) {
    var newMatrix = new SparseMatrix(matrix);
    return newMatrix.%name%();
})
`;

var operators = [
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
    ['>>>', 'rightShift', 'zeroFillRightShift']
];

for (var operator of operators) {
    for (var i = 1; i < operator.length; i++) {
        SparseMatrix.prototype[operator[i]] = eval(fillTemplateFunction(inplaceOperator, {name: operator[i], op: operator[0]}));
        SparseMatrix.prototype[operator[i] + 'S'] = eval(fillTemplateFunction(inplaceOperatorScalar, {name: operator[i] + 'S', op: operator[0]}));
        SparseMatrix.prototype[operator[i] + 'M'] = eval(fillTemplateFunction(inplaceOperatorMatrix, {name: operator[i] + 'M', op: operator[0]}));

        SparseMatrix[operator[i]] = eval(fillTemplateFunction(staticOperator, {name: operator[i]}));
    }
}

var methods = [
    ['~', 'not']
];

[
    'abs', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'cbrt', 'ceil',
    'clz32', 'cos', 'cosh', 'exp', 'expm1', 'floor', 'fround', 'log', 'log1p',
    'log10', 'log2', 'round', 'sign', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc'
].forEach(function (mathMethod) {
    methods.push(['Math.' + mathMethod, mathMethod]);
});

for (var method of methods) {
    for (var i = 1; i < method.length; i++) {
        SparseMatrix.prototype[method[i]] = eval(fillTemplateFunction(inplaceMethod, {name: method[i], method: method[0]}));
        SparseMatrix[method[i]] = eval(fillTemplateFunction(staticMethod, {name: method[i]}));
    }
}

function fillTemplateFunction(template, values) {
    for (var i in values) {
        template = template.replace(new RegExp('%' + i + '%', 'g'), values[i]);
    }
    return template;
}

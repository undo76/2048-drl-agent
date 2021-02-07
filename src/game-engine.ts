export type CellType = { id: number; value: number } | null;

export enum Direction {
    UP = "UP",
    DOWN = "DOWN",
    LEFT = "LEFT",
    RIGHT = "RIGHT",
}

enum Traversal {
    Forward,
    Backwards,
}

enum Axis {
    Horizontal,
    Vertical,
}

function getRandomInt(min: number, max: number): number {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

export class Game {
    over = true;
    score = 0;
    private board!: CellType[][];
    private id = 1;
    private deltaScore: number;
    private invalid = false;

    constructor(public size: number) {
        this.clearBoard();
    }

    private forEachCell(fn: (i: number, j: number) => any) {
        for (let i = 0; i < this.size; i++) {
            for (let j = 0; j < this.size; j++) {
                fn(i, j);
            }
        }
    }

    get cells(): CellType[] {
        return Array.prototype.concat([], ...this.rows);
    }

    private get rows() {
        return this.board;
    }

    private clearBoard() {
        this.over = false;
        this.score = 0;
        this.deltaScore = 0;
        this.invalid = false;
        this.board = new Array(this.size);
        for (var i = 0; i < this.size; i++) {
            this.board[i] = new Array(this.size);
            for (var j = 0; j < this.size; j++) {
                this.setCell(i, j, null);
            }
        }
    }

    private getCell = (i: number, j: number) => {
        return this.board[i][j];
    };

    private setCell = (i: number, j: number, value: CellType): CellType => {
        return (this.board[i][j] = value);
    };

    private transposeFn = <T>(
        f: (first: number, second: number, ...rest: any[]) => T
    ): typeof f => {
        return (first, second, ...rest) => {
            return f(second, first, ...rest);
        };
    };

    private backwardsFn = <T>(
        f: (first: number, second: number, ...rest: any[]) => T
    ): typeof f => {
        return (first, second, ...rest) => {
            return f(first, this.size - second - 1, ...rest);
        };
    };

    private collapse = (i: number, g: any, s: any) => {
        let changed = false;
        let repeat: boolean;

        do {
            repeat = false;
            for (let j = 0; j < this.size - 1; j++) {
                if (g(i, j) === null && g(i, j + 1) !== null) {
                    s(i, j, g(i, j + 1));
                    s(i, j + 1, null);
                    repeat = true;
                    changed = true;
                }
            }
        } while (repeat);
        return changed;
    };

    private move = (axis: Axis, traversal: Traversal, test = false) => {
        let g = this.getCell;
        let s = this.setCell;

        if (axis === Axis.Vertical) {
            g = this.transposeFn(g);
            s = this.transposeFn(s);
        }

        if (traversal === Traversal.Backwards) {
            g = this.backwardsFn(g);
            s = this.backwardsFn(s);
        }

        let changed = false;
        for (let i = 0; i < this.size; i++) {
            changed = this.collapse(i, g, s) || changed;
            for (let j = 0; j < this.size - 1; j++) {
                const current = g(i, j);
                const next = g(i, j + 1);
                if (
                    current !== null &&
                    next !== null &&
                    current.value === next.value
                ) {
                    if (test) return true;
                    const sum = current.value + next.value;
                    this.score += sum;
                    s(i, j, { id: current.id, value: sum });
                    s(i, j + 1, null);
                    changed = true;
                }
            }
            if (!test) {
                changed = this.collapse(i, g, s) || changed;
            }
        }
        return !test && changed;
    };

    private emptyCellsCount = () => {
        let count = 0;
        this.forEachCell((i, j) => {
            const cell = this.getCell(i, j);
            if (cell === null) {
                count++;
            }
        });
        return count;
    };

    private createCell = () => {
        let count = this.emptyCellsCount();
        let idx = getRandomInt(1, count);
        this.forEachCell((i, j) => {
            const cell = this.getCell(i, j);
            if (cell === null) {
                idx--;
                if (idx === 0) {
                    this.setCell(i, j, {
                        id: this.id++,
                        value: getRandomInt(1, 10) === 1 ? 4 : 2,
                    });
                    count--;
                }
            }
        });

        if (this.gameOver(count)) {
            this.over = true;
        }
    };

    private gameOver(emptyCells: number) {
        return (
            emptyCells === 0 &&
            !(
                this.move(Axis.Vertical, Traversal.Forward, true) ||
                this.move(Axis.Horizontal, Traversal.Forward, true)
            )
        );
    }

    reset() {
        this.clearBoard();
        this.createCell();
        return this;
    }

    turn(direction: Direction) {
        let changed = false;
        const prevScore = this.score;

        switch (direction) {
            case Direction.UP:
                changed = this.move(Axis.Vertical, Traversal.Forward);
                break;
            case Direction.DOWN:
                changed = this.move(Axis.Vertical, Traversal.Backwards);
                break;
            case Direction.LEFT:
                changed = this.move(Axis.Horizontal, Traversal.Forward);
                break;
            case Direction.RIGHT:
                changed = this.move(Axis.Horizontal, Traversal.Backwards);
                break;
        }

        if (changed) {
            this.createCell();
        }
        this.invalid = !changed;
        this.deltaScore = this.score - prevScore;
        return changed;
    }

    toString() {
        return this.rows
            .map((row) =>
                row
                    .map((c) => (c ? c.value + "" : "_").padStart(5, " "))
                    .join("  ")
            )
            .join("\n");
    }

    toJson(): string {
        return JSON.stringify({
            board: this.board,
            score: this.score,
            deltaScore: this.deltaScore,
            done: this.over,
            invalid: this.invalid,
            // humanBoard: this.toString()
            humanBoard: "NA"
        }, null, 2);
    }
}

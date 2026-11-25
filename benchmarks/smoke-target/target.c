#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static volatile int sink = 0;

__attribute__((noinline))
static int path_a(const uint8_t *buf, size_t len) {
    int score = 0;
    if (len > 1 && buf[1] == 'A') {
        score += 1;
        if (len > 2 && buf[2] == '1') {
            score += 2;
        }
    }
    sink += score;
    return score;
}

__attribute__((noinline))
static int path_b(const uint8_t *buf, size_t len) {
    int score = 0;
    if (len > 1 && buf[1] == 'B') {
        score += 10;
        if (len > 2 && buf[2] == '2') {
            score += 20;
            if (len > 3 && buf[3] == '!') {
                /* TARGET_LINE_1: Deep target inside path_b */
                sink += 100;
                score += 100;
            } else {
                score += 5;
            }
        }
    }
    sink += score;
    return score;
}

__attribute__((noinline))
static int path_c(const uint8_t *buf, size_t len) {
    int score = 0;
    if (len > 1 && buf[1] == 'C') {
        score += 3;
        if (len > 2 && buf[2] == 'X') {
            if (len > 3 && buf[3] == 'Y') {
                /* TARGET_LINE_2: Alternate target inside path_c */
                sink += 200;
                score += 200;
            }
        }
    }
    sink += score;
    return score;
}

int main(int argc, char **argv) {
    uint8_t buf[256];
    size_t len = 0;
    FILE *fp = stdin;
    if (argc > 1) {
        fp = fopen(argv[1], "rb");
        if (!fp) return 1;
    }
    len = fread(buf, 1, sizeof(buf), fp);
    if (argc > 1) fclose(fp);

    if (len == 0) return 0;

    int res = 0;
    if (buf[0] == 'A') {
        res = path_a(buf, len);
    } else if (buf[0] == 'B') {
        res = path_b(buf, len);
    } else if (buf[0] == 'C') {
        res = path_c(buf, len);
    } else {
        res = -1;
    }
    sink += res;
    return 0;
}

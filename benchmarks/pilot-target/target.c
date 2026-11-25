#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile int g_sink = 0;

__attribute__((noinline))
static int noise_branch(const uint8_t *buf, size_t len) {
    int acc = 0;
    if (len > 2 && buf[1] == 'O') {
        acc += 1;
        if (len > 3 && buf[2] == 'I') {
            acc += 2;
            if (len > 4 && buf[3] == 'S') {
                acc += 3;
                if (len > 5 && buf[4] == 'E') {
                    acc += 4;
                    if (len > 6 && buf[5] == '1') acc += 5;
                    else if (len > 6 && buf[5] == '2') acc += 6;
                }
            }
        }
    } else if (len > 2 && buf[1] == 'E') {
        acc += 7;
        if (len > 3 && buf[2] == 'C') acc += 8;
        if (len > 4 && buf[3] == 'O') acc += 9;
        if (len > 5 && buf[4] == 'Y') acc += 10;
    }
    g_sink += acc;
    return acc;
}

__attribute__((noinline))
static int parse_packet(const uint8_t *buf, size_t len) {
    if (len < 4) return -1;
    uint8_t ver = buf[2];
    uint8_t flags = buf[3];
    int score = 0;

    if (ver == 0x01 || ver == '1') {
        score += 10;
        if (len >= 6 && (flags & 0x03) == 0x01) {
            uint8_t tag = buf[4];
            uint8_t sub = buf[5];
            if (tag == 'T') {
                score += 15;
                if (len >= 8 && sub == 'Z' && buf[6] == '!') {
                    /* TARGET_1: Packet deep TLV branch */
                    g_sink += 101;
                    score += 101;
                    if (buf[7] == 'A') score += 11;
                    else if (buf[7] == 'B') score += 12;
                } else if (len >= 7 && sub == 'Q') {
                    score += 5;
                }
            } else if (tag == 'M') {
                score += 8;
                if (len >= 7 && sub == 'K') score += 9;
            }
        }
    } else if (ver == '2') {
        score += 20;
        int state = 0;
        size_t pos = 4;
        for (int step = 0; step < 3 && pos + 1 < len; ++step, pos += 2) {
            uint8_t op = buf[pos];
            uint8_t arg = buf[pos + 1];
            if (state == 0 && op == 'S' && arg == '1') {
                state = 1;
                score += 12;
            } else if (state == 1 && op == 'S' && arg == '2') {
                state = 2;
                score += 18;
            } else if (state == 2 && op == 'G' && arg == 'O') {
                /* TARGET_2: Stateful multi-record packet sequence */
                g_sink += 202;
                score += 202;
                if (pos + 2 < len && buf[pos + 2] == '#') score += 17;
                break;
            } else if (op == 'X') {
                score += 3;
            }
        }
    }
    g_sink += score;
    return score;
}

__attribute__((noinline))
static int parse_command(const uint8_t *buf, size_t len) {
    if (len < 6) return -1;
    int score = 0;

    if (buf[2] == 'S' && buf[3] == 'E' && buf[4] == 'T') {
        score += 5;
        if (len >= 8 && buf[5] == ':' && buf[6] == '0') score += 6;
        if (len >= 8 && buf[5] == ':' && buf[6] == '1') score += 7;
    } else if (buf[2] == 'A' && buf[3] == 'U' && buf[4] == 'T') {
        score += 14;
        if (len >= 8 && buf[5] == 'H' && buf[6] == ':') {
            score += 16;
            if (len >= 10 && buf[7] == '9' && buf[8] == '!' && buf[9] == '#') {
                /* TARGET_3: Privileged AUTH command branch */
                g_sink += 303;
                score += 303;
                if (len >= 11 && buf[10] == '+') score += 19;
            } else if (buf[7] == '1') {
                score += 4;
            }
        }
    } else if (buf[2] == 'E' && buf[3] == 'X' && buf[4] == 'E') {
        score += 15;
        if (len >= 9 && buf[5] == 'C' && buf[6] == ':') {
            score += 20;
            if (buf[7] == 'R' && buf[8] == 'U') {
                score += 25;
                if (len >= 11 && buf[9] == 'N' && buf[10] == '!') {
                    /* TARGET_4: EXEC:RUN! command branch */
                    g_sink += 404;
                    score += 404;
                    if (len >= 12 && buf[11] == '*') score += 23;
                }
            }
        }
    }
    g_sink += score;
    return score;
}

__attribute__((noinline))
static int run_vm(const uint8_t *buf, size_t len) {
    if (len < 5) return -1;
    int acc = 0;
    int depth = 0;
    int bonus = 0;

    for (size_t i = 2; i < len && i < 16; ++i) {
        uint8_t ins = buf[i];
        if (ins == 'A') {
            acc += 13;
            depth++;
        } else if (ins == 'B') {
            acc += 29;
            depth++;
        } else if (ins == 'C') {
            acc ^= 0x17;
        } else if (ins == 'D') {
            if (depth > 0) depth--;
            acc += 7;
        } else if (ins == 'Z') {
            if (acc == 55 && depth >= 2) {
                /* TARGET_5: VM accumulator & stack invariant */
                g_sink += 505;
                bonus += 505;
                if (i + 1 < len && buf[i + 1] == '!') bonus += 31;
            } else if (acc > 70 && depth >= 3) {
                /* TARGET_6: Alternate VM deep state */
                g_sink += 606;
                bonus += 606;
                if (i + 1 < len && buf[i + 1] == '?') bonus += 37;
            } else {
                bonus += 2;
            }
        }
    }
    g_sink += acc + bonus;
    return acc + bonus;
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

    if (len < 2) return 0;

    int res = 0;
    if (buf[0] == 'P' && buf[1] == 'K') {
        res = parse_packet(buf, len);
    } else if (buf[0] == 'C' && buf[1] == 'M') {
        res = parse_command(buf, len);
    } else if (buf[0] == 'V' && buf[1] == 'M') {
        res = run_vm(buf, len);
    } else if (buf[0] == 'N' || buf[0] == 'D') {
        res = noise_branch(buf, len);
    } else {
        res = -1;
    }
    g_sink += res;
    return 0;
}

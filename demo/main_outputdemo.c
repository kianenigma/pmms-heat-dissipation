#include "output.h"

int main(void)
{
    struct parameters p = { 10, 10, 200, 0.001, 0, 0, 0, 0, 0 };
    struct results r = { 123, -10, 90, 0.00345, 21.4534, 18.5 };
    report_results(&p, &r);
    return 0;
}

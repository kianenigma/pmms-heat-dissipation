#include "output.h"

int main(void)
{
    int f, i;

    for (f = 1; f < 10; ++f) 
    {
        begin_picture(f, 10, 10, -30, 30);

        for (i = 0; i < f; ++i)
            draw_point(i, i, 20.5);
        draw_point(10-f, 0, -10.5);
        draw_point(0, 10-f, f*5.);
        
        end_picture();
    }

    return 0;
}

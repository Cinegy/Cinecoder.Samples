#ifndef __DIB_DRAW_H__
#define __DIB_DRAW_H__

namespace dib_draw
{
	
//-----------------------------------------------------------------------------
template<typename PIXTYPE>
inline void PutPixelToDIB(PIXTYPE *pBMP, int width, int height, int x, int y, PIXTYPE color)
//-----------------------------------------------------------------------------
{
  if(unsigned(x) < unsigned(width) && unsigned(y) < unsigned(height))
    pBMP[y*width+x] = color;
}

//--------------------------------------------------------------
template<typename PIXTYPE>
inline void DrawLine(PIXTYPE *pBMP, int width, int height, int x1, int y1, int x2, int y2, PIXTYPE color)
//--------------------------------------------------------------
{
  int X = x1 << 16, Y = y1 << 16, dx = abs(x2 - x1), dy = abs(y2 - y1), length;                       

  if(!dx && !dy)
  {
    PutPixelToDIB(pBMP, width, height, x1, y1, color);
    return;
  }
  
  if(dx > dy)
  {
    length = dx + 1;
    dy = ((y2 - y1) << 16) / dx;
    dx = x2 > x1 ? (1 << 16) : (-1 << 16);
  }
  else
  {
    length = dy + 1;
    dx = ((x2 - x1) << 16) / dy;
    dy = y2 > y1 ? (1 << 16) : (-1 << 16);
  }

  while(length--)
  {
    PutPixelToDIB(pBMP, width, height, (X+0x8000) >> 16, (Y+0x8000) >> 16, color);
    X += dx;
    Y += dy;
  }

  X-=dx; Y-=dy; // step back
  PutPixelToDIB(pBMP, width, height, (X-dy+0x8000) >> 16, (Y+dx+0x8000) >> 16, color);
  PutPixelToDIB(pBMP, width, height, (X+dy+0x8000) >> 16, (Y-dx+0x8000) >> 16, color);
}

#if 0
//-----------------------------------------------------------------------------
SIZE GetTextSize(const char *s, int scale = 1)
//-----------------------------------------------------------------------------
{
  int textlen = (int)strlen(s);

  SIZE sz;
  sz.cx = textlen * 4 * scale;
  sz.cy = 5 * scale;

  return sz;
}
#endif

//-----------------------------------------------------------------------------
template<typename PIXTYPE>
inline PIXTYPE* GetTextPtr(PIXTYPE *pBMP, int x, int y, int width, int scale)
//-----------------------------------------------------------------------------
{
  return pBMP + (y*width + x) * scale;
}

//-----------------------------------------------------------------------------
template<typename PIXTYPE>
inline void PrintCharToDIB_font4x5(PIXTYPE *pBMP, int x, int y, int width, int c, PIXTYPE forecolor = ~0, PIXTYPE backcolor = 0, int transparency = 0, int scale = 1)
//-----------------------------------------------------------------------------
{
  static char *chars[]   = { "      *   ** * *  ** *    *    *   * *                           ***   * *** *** * * ***  ** *** *** ***                     **  **   ** ***  ** **  *** ***  ** * * *** *** * * *   * * **   *  **   *  **   ** *** * * * * * * * * * * ***  **     **   *       *  ** ",
                             "      *   ** *** **    * * *   *  *   *  * *  *                * * *   *   *   * * * *   *     * * * * *  *   *   *  ***  *    * *** * * * * *   * * *   *   *   * *  *    * * * *   *** * * * * * * * * * * *    *  * * * * * * * * * *   *  *  *    *  * *       * *  ",
                             "      *      * *  *   *   *       *   *   *  ***     ***      *  * *   * ***  ** *** *** ***   * *** ***         *         *  *  * * *** **  *   * * **  **  *   ***  *    * **  *   * * * * * * **  * * **   *   *  * * * * ***  *   *   *   *   *   *               * ",
                             "             ***  ** *   * *      *   *  * *  *   *          *   * *   * *     *   *   * * *   * * *   *  *   *   *  ***  *      **  * * * * *   * * *   *   * * * *  *  * * * * *   * * * * * * *   * * * *   *  *  * * * * *** * *  *  *    *    *  *              *  ",
                             "      *      * * **    * ** *      * *           *       *       ***   * *** ***   * *** ***   * *** **      *                *   ** * * ***  ** **  *** *    ** * * ***  *  * * *** * * * *  *  *    ** * * **   *   **  *   *  * *  *  ***  **     **      ***        ", };

// !"#$%&'()*+,-./  :;<=>?@   Z[\]^_`a   z{|}~

  if((c < ' ' || c > 'z') && c != 127)
    return;

  if(c >= 'a' && c <= 'z')
    c -= 'a' - 'A';

  if(c == 127)
    c = '`' + 1;

  c -= ' ';

  char **ptr = chars;

  pBMP += (y * width + x) * scale;

  for(int j = 0; j < 5; j++)
  {
    PIXTYPE *pbmp = pBMP + (j * width * scale);
                                  
    for(int i = 0; i < 4; i++, pbmp += scale)
    {
      unsigned scaled_color = (ptr[j][c*4+i] == '*' ? forecolor : backcolor) * (255 - transparency);

      for(int yy = 0; yy < scale; yy++) for(int xx = 0; xx < scale; xx++)
        pbmp[yy*width+xx] = PIXTYPE((pbmp[yy*width+xx] * transparency + scaled_color) / 255);
    }
  }
}

//-----------------------------------------------------------------------------
template<typename PIXTYPE>
inline void PrintStringToDIB_font4x5(PIXTYPE *pBMP, int x, int y, int width, const char *s, PIXTYPE forecolor = ~0, PIXTYPE backcolor = 0, int transparency = 0, int scale = 1)
//-----------------------------------------------------------------------------
{
  x -= 4;
  while(*s) PrintCharToDIB_font4x5(pBMP, x+=4, y, width, *s++, forecolor, backcolor, transparency, scale);
}

//----------------------------------------------------------------
template<typename PIXTYPE>
inline void PrintCharToDIB_font8x16(PIXTYPE *pBMP, int x, int y, int width, int c, PIXTYPE forecolor = ~0, PIXTYPE backcolor = 0, int transparency = 0, int scale = 1)
//----------------------------------------------------------------
{
  static unsigned char font[256][16] = {
                                #include "vga@font.inc"
                              };


  pBMP += (y * width + x) * scale;

  for(int j = 0; j < 16; j++)
  {
    PIXTYPE *pbmp = pBMP + (j * width * scale);
                                  
    for(int x = 0x80; x; x>>=1, pbmp += scale)
    {
      unsigned scaled_color = ((font[c][j] & x) ? forecolor : backcolor) * (255 - transparency);

      for(int yy = 0; yy < scale; yy++) for(int xx = 0; xx < scale; xx++)
        pbmp[yy*width+xx] = PIXTYPE((pbmp[yy*width+xx] * transparency + scaled_color) / 255);
    }
  }
}

//-----------------------------------------------------------------------------
template<typename PIXTYPE>
inline void PrintStringToDIB_font8x16(PIXTYPE *pBMP, int x, int y, int width, const char *s, PIXTYPE forecolor = ~0, PIXTYPE backcolor = 0, int transparency = 0, int scale = 1)
//-----------------------------------------------------------------------------
{
  x -= 8;
  while(*s) PrintCharToDIB_font8x16(pBMP, x+=8, y, width, *s++, forecolor, backcolor, transparency, scale);
}


}; // end of namespace

#endif

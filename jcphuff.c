/*
 * jcphuff.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1995-1997, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2011, 2015, 2018, 2021-2022, 2024, D. R. Commander.
 * Copyright (C) 2016, 2018, 2022, Matthieu Darbois.
 * Copyright (C) 2020, Arm Limited.
 * Copyright (C) 2021, Alex Richardson.
 * Copyright (C) 2014, Mozilla Corporation.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains Huffman entropy encoding routines for progressive JPEG.
 *
 * We do not support output suspension in this module, since the library
 * currently does not allow multiple-scan files to be written with output
 * suspension.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#ifdef WITH_SIMD
#include "jsimd.h"
#else
#include "jchuff.h"             /* Declarations shared with jc*huff.c */
#endif
#include <limits.h>

#ifdef HAVE_INTRIN_H
#include <intrin.h>
#ifdef _MSC_VER
#ifdef HAVE_BITSCANFORWARD64
#pragma intrinsic(_BitScanForward64)
#endif
#ifdef HAVE_BITSCANFORWARD
#pragma intrinsic(_BitScanForward)
#endif
#endif
#endif

#ifdef C_PROGRESSIVE_SUPPORTED

#include "jpeg_nbits.h"

/* The progressive Huffman encoder uses the same bit accumulation scheme as
 * the baseline Huffman encoder in jchuff.c, accumulating as many bits as
 * possible in the bit buffer and flushing it in machine-word-sized chunks.
 */

#if defined(__x86_64__) && defined(__ILP32__)
typedef unsigned long long bit_buf_type;
#else
typedef size_t bit_buf_type;
#endif

#if (defined(SIZEOF_SIZE_T) && SIZEOF_SIZE_T == 8) || defined(_WIN64) || \
    (defined(__x86_64__) && defined(__ILP32__))
#define BIT_BUF_SIZE  64
#elif (defined(SIZEOF_SIZE_T) && SIZEOF_SIZE_T == 4) || defined(_WIN32)
#define BIT_BUF_SIZE  32
#else
#error Cannot determine word size
#endif


/* Expanded entropy encoder object for progressive Huffman encoding. */

typedef struct {
  struct jpeg_entropy_encoder pub; /* public fields */

  /* Pointer to routine to prepare data for encode_mcu_AC_first() */
  void (*AC_first_prepare) (const JCOEF *block,
                            const int *jpeg_natural_order_start, int Sl,
                            int Al, UJCOEF *values, size_t *zerobits);
  /* Pointer to routine to prepare data for encode_mcu_AC_refine() */
  int (*AC_refine_prepare) (const JCOEF *block,
                            const int *jpeg_natural_order_start, int Sl,
                            int Al, UJCOEF *absvalues, size_t *bits);

  /* Mode flag: TRUE for optimization, FALSE for actual data output */
  boolean gather_statistics;

  /* Bit-level coding status.
   * next_output_byte/free_in_buffer are local copies of cinfo->dest fields.
   */
  JOCTET *next_output_byte;     /* => next byte to write in buffer */
  size_t free_in_buffer;        /* # of byte spaces remaining in buffer */
  bit_buf_type put_buffer;      /* current bit-accumulation buffer */
  int free_bits;                /* # of bits available in it */
  j_compress_ptr cinfo;         /* link to cinfo (needed for dump_buffer) */

  /* Coding status for DC components */
  int last_dc_val[MAX_COMPS_IN_SCAN]; /* last DC coef for each component */

  /* Coding status for AC components */
  int ac_tbl_no;                /* the table number of the single component */
  unsigned int EOBRUN;          /* run length of EOBs */
  unsigned int BE;              /* # of buffered correction bits before MCU */
  char *bit_buffer;             /* buffer for correction bits (1 per char) */
  /* packing correction bits tightly would save some space but cost time... */

  unsigned int restarts_to_go;  /* MCUs left in this restart interval */
  int next_restart_num;         /* next restart number to write (0-7) */

  /* Pointers to derived tables (these workspaces have image lifespan).
   * Since any one scan codes only DC or only AC, we only need one set
   * of tables, not one for DC and one for AC.
   */
  c_derived_tbl *derived_tbls[NUM_HUFF_TBLS];

  /* Statistics tables for optimization; again, one set is enough */
  long *count_ptrs[NUM_HUFF_TBLS];
} phuff_entropy_encoder;

typedef phuff_entropy_encoder *phuff_entropy_ptr;

/* MAX_CORR_BITS is the number of bits the AC refinement correction-bit
 * buffer can hold.  Larger sizes may slightly improve compression, but
 * 1000 is already well into the realm of overkill.
 * The minimum safe size is 64 bits.
 */

#define MAX_CORR_BITS  1000     /* Max # of correction bits I can buffer */

/* IRIGHT_SHIFT is like RIGHT_SHIFT, but works on int rather than JLONG.
 * We assume that int right shift is unsigned if JLONG right shift is,
 * which should be safe.
 */

#ifdef RIGHT_SHIFT_IS_UNSIGNED
#define ISHIFT_TEMPS    int ishift_temp;
#define IRIGHT_SHIFT(x, shft) \
  ((ishift_temp = (x)) < 0 ? \
   (ishift_temp >> (shft)) | ((~0) << (16 - (shft))) : \
   (ishift_temp >> (shft)))
#else
#define ISHIFT_TEMPS
#define IRIGHT_SHIFT(x, shft)   ((x) >> (shft))
#endif

#define PAD(v, p)  ((v + (p) - 1) & (~((p) - 1)))

/* Forward declarations */
METHODDEF(boolean) encode_mcu_DC_first(j_compress_ptr cinfo,
                                       JBLOCKROW *MCU_data);
METHODDEF(void) encode_mcu_AC_first_prepare
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *values, size_t *zerobits);
METHODDEF(boolean) encode_mcu_AC_first(j_compress_ptr cinfo,
                                       JBLOCKROW *MCU_data);
METHODDEF(boolean) encode_mcu_DC_refine(j_compress_ptr cinfo,
                                        JBLOCKROW *MCU_data);
METHODDEF(int) encode_mcu_AC_refine_prepare
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *absvalues, size_t *bits);
METHODDEF(boolean) encode_mcu_AC_refine(j_compress_ptr cinfo,
                                        JBLOCKROW *MCU_data);
METHODDEF(void) finish_pass_phuff(j_compress_ptr cinfo);
METHODDEF(void) finish_pass_gather_phuff(j_compress_ptr cinfo);


/* Count bit loop zeroes */
INLINE
METHODDEF(int)
count_zeroes(size_t *x)
{
#if defined(HAVE_BUILTIN_CTZL)
  int result;
  result = __builtin_ctzl(*x);
  *x >>= result;
#elif defined(HAVE_BITSCANFORWARD64)
  unsigned long result;
  _BitScanForward64(&result, *x);
  *x >>= result;
#elif defined(HAVE_BITSCANFORWARD)
  unsigned long result;
  _BitScanForward(&result, *x);
  *x >>= result;
#else
  int result = 0;
  while ((*x & 1) == 0) {
    ++result;
    *x >>= 1;
  }
#endif
  return (int)result;
}


/*
 * Initialize for a Huffman-compressed scan using progressive JPEG.
 */

METHODDEF(void)
start_pass_phuff(j_compress_ptr cinfo, boolean gather_statistics)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  boolean is_DC_band;
  int ci, tbl;
  jpeg_component_info *compptr;

  entropy->cinfo = cinfo;
  entropy->gather_statistics = gather_statistics;

  is_DC_band = (cinfo->Ss == 0);

  /* We assume jcmaster.c already validated the scan parameters. */

  /* Select execution routines */
  if (cinfo->Ah == 0) {
    if (is_DC_band)
      entropy->pub.encode_mcu = encode_mcu_DC_first;
    else
      entropy->pub.encode_mcu = encode_mcu_AC_first;
#ifdef WITH_SIMD
    if (jsimd_can_encode_mcu_AC_first_prepare())
      entropy->AC_first_prepare = jsimd_encode_mcu_AC_first_prepare;
    else
#endif
      entropy->AC_first_prepare = encode_mcu_AC_first_prepare;
  } else {
    if (is_DC_band)
      entropy->pub.encode_mcu = encode_mcu_DC_refine;
    else {
      entropy->pub.encode_mcu = encode_mcu_AC_refine;
#ifdef WITH_SIMD
      if (jsimd_can_encode_mcu_AC_refine_prepare())
        entropy->AC_refine_prepare = jsimd_encode_mcu_AC_refine_prepare;
      else
#endif
        entropy->AC_refine_prepare = encode_mcu_AC_refine_prepare;
      /* AC refinement needs a correction bit buffer */
      if (entropy->bit_buffer == NULL)
        entropy->bit_buffer = (char *)
          (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                      MAX_CORR_BITS * sizeof(char));
    }
  }
  if (gather_statistics)
    entropy->pub.finish_pass = finish_pass_gather_phuff;
  else
    entropy->pub.finish_pass = finish_pass_phuff;

  /* Only DC coefficients may be interleaved, so cinfo->comps_in_scan = 1
   * for AC coefficients.
   */
  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    /* Initialize DC predictions to 0 */
    entropy->last_dc_val[ci] = 0;
    /* Get table index */
    if (is_DC_band) {
      if (cinfo->Ah != 0)       /* DC refinement needs no table */
        continue;
      tbl = compptr->dc_tbl_no;
    } else {
      entropy->ac_tbl_no = tbl = compptr->ac_tbl_no;
    }
    if (gather_statistics) {
      /* Check for invalid table index */
      /* (make_c_derived_tbl does this in the other path) */
      if (tbl < 0 || tbl >= NUM_HUFF_TBLS)
        ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, tbl);
      /* Allocate and zero the statistics tables */
      /* Note that jpeg_gen_optimal_table expects 257 entries in each table! */
      if (entropy->count_ptrs[tbl] == NULL)
        entropy->count_ptrs[tbl] = (long *)
          (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                      257 * sizeof(long));
      memset(entropy->count_ptrs[tbl], 0, 257 * sizeof(long));

      if (cinfo->master->trellis_passes) {
        /* When generating tables for trellis passes, make sure that all */
        /* codewords have an assigned length */
        int i, j;
        for (i = 0; i < 16; i++)
          for (j = 0; j < 12; j++)
            entropy->count_ptrs[tbl][16 * i + j] = 1;
      }
    } else {
      /* Compute derived values for Huffman table */
      /* We may do this more than once for a table, but it's not expensive */
      jpeg_make_c_derived_tbl(cinfo, is_DC_band, tbl,
                              &entropy->derived_tbls[tbl]);
    }
  }

  /* Initialize AC stuff */
  entropy->EOBRUN = 0;
  entropy->BE = 0;

  /* Initialize bit buffer to empty */
  entropy->put_buffer = 0;
  entropy->free_bits = BIT_BUF_SIZE;

  /* Initialize restart stuff */
  entropy->restarts_to_go = cinfo->restart_interval;
  entropy->next_restart_num = 0;
}


/* Outputting bytes to the file.
 * NB: these must be called only when actually outputting,
 * that is, entropy->gather_statistics == FALSE.
 */

/* Emit a byte */
#define emit_byte(entropy, val) { \
  *(entropy)->next_output_byte++ = (JOCTET)(val); \
  if (--(entropy)->free_in_buffer == 0) \
    dump_buffer(entropy); \
}


LOCAL(void)
dump_buffer(phuff_entropy_ptr entropy)
/* Empty the output buffer; we do not support suspension in this module. */
{
  struct jpeg_destination_mgr *dest = entropy->cinfo->dest;

  if (!(*dest->empty_output_buffer) (entropy->cinfo))
    ERREXIT(entropy->cinfo, JERR_CANT_SUSPEND);
  /* After a successful buffer dump, must reset buffer pointers */
  entropy->next_output_byte = dest->next_output_byte;
  entropy->free_in_buffer = dest->free_in_buffer;
}


/* Outputting bits to the file */

/* Output the entire bit buffer.  If there are no 0xFF bytes in it, then write
 * it to the output buffer directly.  Otherwise, encode 0xFF as 0xFF 0x00 one
 * byte at a time.  (Same scheme as the baseline Huffman encoder in jchuff.c,
 * adapted to this module's destination management.)
 */

LOCAL(void)
flush_put_buffer(phuff_entropy_ptr entropy, bit_buf_type put_buffer)
{
#if BIT_BUF_SIZE == 64
  if (entropy->free_in_buffer > 8 &&
      !(put_buffer & 0x8080808080808080 &
        ~(put_buffer + 0x0101010101010101))) {
    JOCTET *buffer = entropy->next_output_byte;
    buffer[0] = (JOCTET)(put_buffer >> 56);
    buffer[1] = (JOCTET)(put_buffer >> 48);
    buffer[2] = (JOCTET)(put_buffer >> 40);
    buffer[3] = (JOCTET)(put_buffer >> 32);
    buffer[4] = (JOCTET)(put_buffer >> 24);
    buffer[5] = (JOCTET)(put_buffer >> 16);
    buffer[6] = (JOCTET)(put_buffer >> 8);
    buffer[7] = (JOCTET)(put_buffer);
    entropy->next_output_byte += 8;
    entropy->free_in_buffer -= 8;
  } else
#else
  if (entropy->free_in_buffer > 4 &&
      !(put_buffer & 0x80808080 & ~(put_buffer + 0x01010101))) {
    JOCTET *buffer = entropy->next_output_byte;
    buffer[0] = (JOCTET)(put_buffer >> 24);
    buffer[1] = (JOCTET)(put_buffer >> 16);
    buffer[2] = (JOCTET)(put_buffer >> 8);
    buffer[3] = (JOCTET)(put_buffer);
    entropy->next_output_byte += 4;
    entropy->free_in_buffer -= 4;
  } else
#endif
  {
    int put_bits = BIT_BUF_SIZE;

    while (put_bits > 0) {
      int c;

      put_bits -= 8;
      c = (int)((put_buffer >> put_bits) & 0xFF);
      emit_byte(entropy, c);
      if (c == 0xFF) {          /* need to stuff a zero byte? */
        emit_byte(entropy, 0);
      }
    }
  }
}


/* Insert code into the bit buffer and output the bit buffer if needed.
 * NOTE: We can't flush with free_bits == 0, since the left shift in
 * the flush path would have undefined behavior.
 */

INLINE
LOCAL(void)
emit_bits(phuff_entropy_ptr entropy, unsigned int code, int size)
/* Emit some bits, unless we are in gather mode */
{
  /* This routine is heavily used, so it's worth coding tightly. */
  bit_buf_type put_buffer;
  int free_bits;

  /* if size is 0, caller used an invalid Huffman table entry */
  if (size == 0)
    ERREXIT(entropy->cinfo, JERR_HUFF_MISSING_CODE);

  if (entropy->gather_statistics)
    return;                     /* do nothing if we're only getting stats */

  put_buffer = entropy->put_buffer;
  free_bits = entropy->free_bits - size;
  code &= (((bit_buf_type)1) << size) - 1; /* mask off any extra bits */

  if (free_bits < 0) {
    /* Fill the bit buffer to capacity with the leading bits from code, then
     * output the bit buffer and put the remaining bits from code into it.
     */
    put_buffer = (put_buffer << (size + free_bits)) | (code >> -free_bits);
    flush_put_buffer(entropy, put_buffer);
    free_bits += BIT_BUF_SIZE;
    put_buffer = code;
  } else {
    put_buffer = (put_buffer << size) | code;
  }

  entropy->put_buffer = put_buffer;
  entropy->free_bits = free_bits;
}


LOCAL(void)
flush_bits(phuff_entropy_ptr entropy)
{
  bit_buf_type put_buffer = entropy->put_buffer;
  int put_bits = BIT_BUF_SIZE - entropy->free_bits;

  while (put_bits >= 8) {
    int c;

    put_bits -= 8;
    c = (int)((put_buffer >> put_bits) & 0xFF);
    emit_byte(entropy, c);
    if (c == 0xFF) {            /* need to stuff a zero byte? */
      emit_byte(entropy, 0);
    }
  }
  if (put_bits > 0) {
    /* fill any partial byte with ones */
    int c = (int)(((put_buffer << (8 - put_bits)) | (0xFF >> put_bits)) & 0xFF);

    emit_byte(entropy, c);
    if (c == 0xFF) {            /* need to stuff a zero byte? */
      emit_byte(entropy, 0);
    }
  }

  entropy->put_buffer = 0;      /* and reset bit buffer to empty */
  entropy->free_bits = BIT_BUF_SIZE;
}


/* The tightest loops (currently only the AC initial-scan encoding loop) keep
 * the bit buffer and the output pointer in local variables and use the same
 * macros as the baseline Huffman encoder in jchuff.c, writing into a local
 * buffer that is guaranteed to be large enough for one MCU's worth of data.
 */

/* Output byte b and, speculatively, an additional 0 byte.  0xFF must be
 * encoded as 0xFF 0x00, so the output buffer pointer is advanced by 2 if the
 * byte is 0xFF.  Otherwise, the output buffer pointer is advanced by 1, and
 * the speculative 0 byte will be overwritten by the next byte.
 */
#define EMIT_BYTE(b) { \
  buffer[0] = (JOCTET)(b); \
  buffer[1] = 0; \
  buffer -= -2 + ((JOCTET)(b) < 0xFF); \
}

/* Output the entire bit buffer.  If there are no 0xFF bytes in it, then write
 * directly to the output buffer.  Otherwise, use the EMIT_BYTE() macro to
 * encode 0xFF as 0xFF 0x00.
 */
#if BIT_BUF_SIZE == 64

#define FLUSH() { \
  if (put_buffer & 0x8080808080808080 & ~(put_buffer + 0x0101010101010101)) { \
    EMIT_BYTE(put_buffer >> 56) \
    EMIT_BYTE(put_buffer >> 48) \
    EMIT_BYTE(put_buffer >> 40) \
    EMIT_BYTE(put_buffer >> 32) \
    EMIT_BYTE(put_buffer >> 24) \
    EMIT_BYTE(put_buffer >> 16) \
    EMIT_BYTE(put_buffer >>  8) \
    EMIT_BYTE(put_buffer      ) \
  } else { \
    buffer[0] = (JOCTET)(put_buffer >> 56); \
    buffer[1] = (JOCTET)(put_buffer >> 48); \
    buffer[2] = (JOCTET)(put_buffer >> 40); \
    buffer[3] = (JOCTET)(put_buffer >> 32); \
    buffer[4] = (JOCTET)(put_buffer >> 24); \
    buffer[5] = (JOCTET)(put_buffer >> 16); \
    buffer[6] = (JOCTET)(put_buffer >> 8); \
    buffer[7] = (JOCTET)(put_buffer); \
    buffer += 8; \
  } \
}

#else

#define FLUSH() { \
  if (put_buffer & 0x80808080 & ~(put_buffer + 0x01010101)) { \
    EMIT_BYTE(put_buffer >> 24) \
    EMIT_BYTE(put_buffer >> 16) \
    EMIT_BYTE(put_buffer >>  8) \
    EMIT_BYTE(put_buffer      ) \
  } else { \
    buffer[0] = (JOCTET)(put_buffer >> 24); \
    buffer[1] = (JOCTET)(put_buffer >> 16); \
    buffer[2] = (JOCTET)(put_buffer >> 8); \
    buffer[3] = (JOCTET)(put_buffer); \
    buffer += 4; \
  } \
}

#endif

/* Fill the bit buffer to capacity with the leading bits from code, then output
 * the bit buffer and put the remaining bits from code into the bit buffer.
 */
#define PUT_AND_FLUSH(code, size) { \
  put_buffer = (put_buffer << (size + free_bits)) | (code >> -free_bits); \
  FLUSH() \
  free_bits += BIT_BUF_SIZE; \
  put_buffer = code; \
}

/* Insert code into the bit buffer and output the bit buffer if needed.
 * NOTE: We can't flush with free_bits == 0, since the left shift in
 * PUT_AND_FLUSH() would have undefined behavior.
 */
#define PUT_BITS(code, size) { \
  free_bits -= size; \
  if (free_bits < 0) \
    PUT_AND_FLUSH(code, size) \
  else \
    put_buffer = (put_buffer << size) | code; \
}

/* Although it is exceedingly rare, it is possible for an encoded MCU (one
 * block in an AC scan) to be larger than the 128-byte unencoded block.  In an
 * AC initial scan, up to 63 coefficients are coded with at most 16 + 14 bits
 * each, so an encoded block cannot be larger than 30 * 63 / 8 bytes plus byte
 * stuffing plus the granularity of bit-buffer flushes.  In an AC refinement
 * scan, an encoded block cannot be larger than an EOB run (16 + 14 bits), up
 * to MAX_CORR_BITS buffered correction bits, and up to 63 newly-nonzero
 * coefficients with at most 16 + 1 bits each -- about 270 bytes, plus byte
 * stuffing plus the granularity of bit-buffer flushes.
 */
#define BUFSIZE  (DCTSIZE2 * 8)
#define BUFSIZE_AC_REFINE  (DCTSIZE2 * 16)

#define LOAD_BUFFER(bufsize) { \
  if (entropy->free_in_buffer < (bufsize)) { \
    localbuf = 1; \
    buffer = _buffer; \
  } else \
    buffer = entropy->next_output_byte; \
}

#define STORE_BUFFER() { \
  if (localbuf) { \
    size_t bytes, bytestocopy; \
    bytes = buffer - _buffer; \
    buffer = _buffer; \
    while (bytes > 0) { \
      bytestocopy = MIN(bytes, entropy->free_in_buffer); \
      memcpy(entropy->next_output_byte, buffer, bytestocopy); \
      entropy->next_output_byte += bytestocopy; \
      buffer += bytestocopy; \
      entropy->free_in_buffer -= bytestocopy; \
      if (entropy->free_in_buffer == 0) \
        dump_buffer(entropy); \
      bytes -= bytestocopy; \
    } \
  } else { \
    entropy->free_in_buffer -= (buffer - entropy->next_output_byte); \
    entropy->next_output_byte = buffer; \
    /* The emit_byte() macro requires at least one free byte in the output \
     * buffer, so empty the buffer now if it is full. \
     */ \
    if (entropy->free_in_buffer == 0) \
      dump_buffer(entropy); \
  } \
}


/*
 * Emit (or just count) a Huffman symbol.
 */

INLINE
LOCAL(void)
emit_symbol(phuff_entropy_ptr entropy, int tbl_no, int symbol)
{
  if (entropy->gather_statistics)
    entropy->count_ptrs[tbl_no][symbol]++;
  else {
    c_derived_tbl *tbl = entropy->derived_tbls[tbl_no];
    emit_bits(entropy, tbl->ehufco[symbol], tbl->ehufsi[symbol]);
  }
}


/*
 * Emit (or just count) a Huffman symbol, followed by nbits of the value, if
 * nonzero.  Concatenating the Huffman code and the value bits allows them to
 * be inserted into the bit buffer with a single emit_bits() call (same trick
 * as the PUT_CODE() macro in jchuff.c.)  The Huffman code is at most 16 bits,
 * and nbits is at most 15, so the combined size always fits in the bit
 * buffer.
 */

INLINE
LOCAL(void)
emit_symbol_and_bits(phuff_entropy_ptr entropy, int tbl_no, int symbol,
                     unsigned int value, int nbits)
{
  if (entropy->gather_statistics)
    entropy->count_ptrs[tbl_no][symbol]++;
  else {
    c_derived_tbl *tbl = entropy->derived_tbls[tbl_no];
    int size = tbl->ehufsi[symbol];

    /* if size is 0, caller used an invalid Huffman table entry */
    if (size == 0)
      ERREXIT(entropy->cinfo, JERR_HUFF_MISSING_CODE);

    value &= (((unsigned int)1) << nbits) - 1; /* mask off any extra bits */
    emit_bits(entropy, (tbl->ehufco[symbol] << nbits) | value, size + nbits);
  }
}


/*
 * Emit bits from a correction bit buffer.
 */

LOCAL(void)
emit_buffered_bits(phuff_entropy_ptr entropy, char *bufstart,
                   unsigned int nbits)
{
  if (entropy->gather_statistics)
    return;                     /* no real work */

  /* Pack groups of 8 correction bits (one per char, in emission order) into
   * a single emit_bits() call.  emit_bits() emits the most significant bit
   * of code first, so this preserves the bit order.
   */
  while (nbits >= 8) {
    unsigned int code =
      ((unsigned int)(bufstart[0] & 1) << 7) |
      ((unsigned int)(bufstart[1] & 1) << 6) |
      ((unsigned int)(bufstart[2] & 1) << 5) |
      ((unsigned int)(bufstart[3] & 1) << 4) |
      ((unsigned int)(bufstart[4] & 1) << 3) |
      ((unsigned int)(bufstart[5] & 1) << 2) |
      ((unsigned int)(bufstart[6] & 1) << 1) |
      ((unsigned int)(bufstart[7] & 1));

    emit_bits(entropy, code, 8);
    bufstart += 8;
    nbits -= 8;
  }
  while (nbits > 0) {
    emit_bits(entropy, (unsigned int)(*bufstart), 1);
    bufstart++;
    nbits--;
  }
}


/*
 * Emit any pending EOBRUN symbol.
 */

LOCAL(void)
emit_eobrun(phuff_entropy_ptr entropy)
{
  register int temp, nbits;

  if (entropy->EOBRUN > 0) {    /* if there is any pending EOBRUN */
    temp = entropy->EOBRUN;
    nbits = JPEG_NBITS_NONZERO(temp) - 1;
    /* safety check: shouldn't happen given limited correction-bit buffer */
    if (nbits > 14)
      ERREXIT(entropy->cinfo, JERR_HUFF_MISSING_CODE);

    emit_symbol_and_bits(entropy, entropy->ac_tbl_no, nbits << 4,
                         entropy->EOBRUN, nbits);

    entropy->EOBRUN = 0;

    /* Emit any buffered correction bits */
    emit_buffered_bits(entropy, entropy->bit_buffer, entropy->BE);
    entropy->BE = 0;
  }
}


/*
 * Emit a restart marker & resynchronize predictions.
 */

LOCAL(void)
emit_restart(phuff_entropy_ptr entropy, int restart_num)
{
  int ci;

  emit_eobrun(entropy);

  if (!entropy->gather_statistics) {
    flush_bits(entropy);
    emit_byte(entropy, 0xFF);
    emit_byte(entropy, JPEG_RST0 + restart_num);
  }

  if (entropy->cinfo->Ss == 0) {
    /* Re-initialize DC predictions to 0 */
    for (ci = 0; ci < entropy->cinfo->comps_in_scan; ci++)
      entropy->last_dc_val[ci] = 0;
  } else {
    /* Re-initialize all AC-related fields to 0 */
    entropy->EOBRUN = 0;
    entropy->BE = 0;
  }
}


/*
 * MCU encoding for DC initial scan (either spectral selection,
 * or first pass of successive approximation).
 */

METHODDEF(boolean)
encode_mcu_DC_first(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  register int temp, temp2, temp3;
  register int nbits;
  int blkn, ci;
  int Al = cinfo->Al;
  JBLOCKROW block;
  jpeg_component_info *compptr;
  ISHIFT_TEMPS
  int max_coef_bits = cinfo->data_precision + 2;

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart(entropy, entropy->next_restart_num);

  /* Encode the MCU data blocks */
  for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
    block = MCU_data[blkn];
    ci = cinfo->MCU_membership[blkn];
    compptr = cinfo->cur_comp_info[ci];

    /* Compute the DC value after the required point transform by Al.
     * This is simply an arithmetic right shift.
     */
    temp2 = IRIGHT_SHIFT((int)((*block)[0]), Al);

    /* DC differences are figured on the point-transformed values. */
    temp = temp2 - entropy->last_dc_val[ci];
    entropy->last_dc_val[ci] = temp2;

    /* Encode the DC coefficient difference per section G.1.2.1 */

    /* This is a well-known technique for obtaining the absolute value without
     * a branch.  It is derived from an assembly language technique presented
     * in "How to Optimize for the Pentium Processors", Copyright (c) 1996,
     * 1997 by Agner Fog.
     */
    temp3 = temp >> (CHAR_BIT * sizeof(int) - 1);
    temp ^= temp3;
    temp -= temp3;              /* temp is abs value of input */
    /* For a negative input, want temp2 = bitwise complement of abs(input) */
    temp2 = temp ^ temp3;

    /* Find the number of bits needed for the magnitude of the coefficient */
    nbits = JPEG_NBITS(temp);
    /* Check for out-of-range coefficient values.
     * Since we're encoding a difference, the range limit is twice as much.
     */
    if (nbits > max_coef_bits + 1)
      ERREXIT(cinfo, JERR_BAD_DCT_COEF);

    /* Count/emit the Huffman-coded symbol for the number of bits, and emit
     * that number of bits of the value, if positive, or the complement of
     * its magnitude, if negative.
     */
    emit_symbol_and_bits(entropy, compptr->dc_tbl_no, nbits,
                         (unsigned int)temp2, nbits);
  }

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;

  /* Update restart-interval state too */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      entropy->restarts_to_go = cinfo->restart_interval;
      entropy->next_restart_num++;
      entropy->next_restart_num &= 7;
    }
    entropy->restarts_to_go--;
  }

  return TRUE;
}


/*
 * Data preparation for encode_mcu_AC_first().
 */

#define COMPUTE_ABSVALUES_AC_FIRST(Sl) { \
  for (k = 0; k < Sl; k++) { \
    temp = block[jpeg_natural_order_start[k]]; \
    if (temp == 0) \
      continue; \
    /* We must apply the point transform by Al.  For AC coefficients this \
     * is an integer division with rounding towards 0.  To do this portably \
     * in C, we shift after obtaining the absolute value; so the code is \
     * interwoven with finding the abs value (temp) and output bits (temp2). \
     */ \
    temp2 = temp >> (CHAR_BIT * sizeof(int) - 1); \
    temp ^= temp2; \
    temp -= temp2;              /* temp is abs value of input */ \
    temp >>= Al;                /* apply the point transform */ \
    /* Watch out for case that nonzero coef is zero after point transform */ \
    if (temp == 0) \
      continue; \
    /* For a negative coef, want temp2 = bitwise complement of abs(coef) */ \
    temp2 ^= temp; \
    values[k] = (UJCOEF)temp; \
    values[k + DCTSIZE2] = (UJCOEF)temp2; \
    zerobits |= ((size_t)1U) << k; \
  } \
}

METHODDEF(void)
encode_mcu_AC_first_prepare(const JCOEF *block,
                            const int *jpeg_natural_order_start, int Sl,
                            int Al, UJCOEF *values, size_t *bits)
{
  register int k, temp, temp2;
  size_t zerobits = 0U;
  int Sl0 = Sl;

#if SIZEOF_SIZE_T == 4
  if (Sl0 > 32)
    Sl0 = 32;
#endif

  COMPUTE_ABSVALUES_AC_FIRST(Sl0);

  bits[0] = zerobits;
#if SIZEOF_SIZE_T == 4
  zerobits = 0U;

  if (Sl > 32) {
    Sl -= 32;
    jpeg_natural_order_start += 32;
    values += 32;

    COMPUTE_ABSVALUES_AC_FIRST(Sl);
  }
  bits[1] = zerobits;
#endif
}

/*
 * MCU encoding for AC initial scan (either spectral selection,
 * or first pass of successive approximation).
 */

/* The body of the per-coefficient encoding loop, parameterized on the
 * operations performed for a run-length-16 (ZRL) code and for a coefficient
 * code, so that the statistics-gathering and the data-output passes can each
 * use a specialized loop.
 */
#define ENCODE_COEFS_AC_FIRST(label, zrl_op, code_op) { \
  while (zerobits) { \
    r = count_zeroes(&zerobits); \
    cvalue += r; \
label \
    temp  = cvalue[0]; \
    temp2 = cvalue[DCTSIZE2]; \
    \
    /* if run length > 15, must emit special run-length-16 codes (0xF0) */ \
    while (r > 15) { \
      zrl_op \
      r -= 16; \
    } \
    \
    /* Find the number of bits needed for the magnitude of the coefficient */ \
    nbits = JPEG_NBITS_NONZERO(temp);  /* there must be at least one 1 bit */ \
    /* Check for out-of-range coefficient values */ \
    if (nbits > max_coef_bits) \
      ERREXIT(cinfo, JERR_BAD_DCT_COEF); \
    \
    /* Count/emit Huffman symbol for run length / number of bits, and emit \
     * that number of bits of the value, if positive, or the complement of \
     * its magnitude, if negative. \
     */ \
    code_op \
    \
    cvalue++; \
    zerobits >>= 1; \
  } \
}

/* Count a ZRL/coefficient symbol (statistics-gathering pass) */

#define COUNT_ZRL_AC_FIRST  { ac_counts[0xF0]++; }

#define COUNT_CODE_AC_FIRST { ac_counts[(r << 4) + nbits]++; }

/* Emit a ZRL/coefficient symbol (data-output pass).  The Huffman code for
 * the symbol and the magnitude bits of the coefficient are concatenated and
 * inserted into the bit buffer with a single PUT_BITS() invocation, as in
 * jchuff.c.
 */

#define EMIT_ZRL_AC_FIRST { \
  if (zrl_size == 0) \
    ERREXIT(cinfo, JERR_HUFF_MISSING_CODE); \
  PUT_BITS(zrl_code, zrl_size) \
}

#define EMIT_CODE_AC_FIRST { \
  symbol = (r << 4) + nbits; \
  size = actbl->ehufsi[symbol]; \
  /* if size is 0, caller used an invalid Huffman table entry */ \
  if (size == 0) \
    ERREXIT(cinfo, JERR_HUFF_MISSING_CODE); \
  code = (actbl->ehufco[symbol] << nbits) | \
         ((unsigned int)temp2 & ((1U << nbits) - 1)); \
  size += nbits; \
  PUT_BITS(code, size) \
}

METHODDEF(boolean)
encode_mcu_AC_first(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  register int temp, temp2;
  register int nbits, r;
  int Sl = cinfo->Se - cinfo->Ss + 1;
  int Al = cinfo->Al;
  UJCOEF values_unaligned[2 * DCTSIZE2 + 15];
  UJCOEF *values;
  const UJCOEF *cvalue;
  size_t zerobits;
  size_t bits[8 / SIZEOF_SIZE_T];
  int max_coef_bits = cinfo->data_precision + 2;

#ifdef ZERO_BUFFERS
  memset(values_unaligned, 0, sizeof(values_unaligned));
  memset(bits, 0, sizeof(bits));
#endif

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart(entropy, entropy->next_restart_num);

#ifdef WITH_SIMD
  cvalue = values = (UJCOEF *)PAD((JUINTPTR)values_unaligned, 16);
#else
  /* Not using SIMD, so alignment is not needed */
  cvalue = values = values_unaligned;
#endif

  /* Prepare data */
  entropy->AC_first_prepare(MCU_data[0][0], jpeg_natural_order + cinfo->Ss,
                            Sl, Al, values, bits);

  zerobits = bits[0];
#if SIZEOF_SIZE_T == 4
  zerobits |= bits[1];
#endif

  /* Emit any pending EOBRUN */
  if (zerobits && (entropy->EOBRUN > 0))
    emit_eobrun(entropy);

#if SIZEOF_SIZE_T == 4
  zerobits = bits[0];
#endif

  /* Encode the AC coefficients per section G.1.2.2, fig. G.3 */

  if (entropy->gather_statistics) {
    long *ac_counts = entropy->count_ptrs[entropy->ac_tbl_no];

    ENCODE_COEFS_AC_FIRST((void)0;, COUNT_ZRL_AC_FIRST, COUNT_CODE_AC_FIRST);

#if SIZEOF_SIZE_T == 4
    zerobits = bits[1];
    if (zerobits) {
      int diff = ((values + DCTSIZE2 / 2) - cvalue);
      r = count_zeroes(&zerobits);
      r += diff;
      cvalue += r;
      goto first_iter_ac_first_gather;
    }

    ENCODE_COEFS_AC_FIRST(first_iter_ac_first_gather:, COUNT_ZRL_AC_FIRST,
                          COUNT_CODE_AC_FIRST);
#endif
  } else {
    c_derived_tbl *actbl = entropy->derived_tbls[entropy->ac_tbl_no];
    unsigned int zrl_code = actbl->ehufco[0xF0];
    int zrl_size = actbl->ehufsi[0xF0];
    unsigned int code;
    int symbol, size;
    JOCTET _buffer[BUFSIZE], *buffer;
    bit_buf_type put_buffer = entropy->put_buffer;
    int free_bits = entropy->free_bits;
    int localbuf = 0;

#ifdef ZERO_BUFFERS
    memset(_buffer, 0, sizeof(_buffer));
#endif

    LOAD_BUFFER(BUFSIZE)

    ENCODE_COEFS_AC_FIRST((void)0;, EMIT_ZRL_AC_FIRST, EMIT_CODE_AC_FIRST);

#if SIZEOF_SIZE_T == 4
    zerobits = bits[1];
    if (zerobits) {
      int diff = ((values + DCTSIZE2 / 2) - cvalue);
      r = count_zeroes(&zerobits);
      r += diff;
      cvalue += r;
      goto first_iter_ac_first_emit;
    }

    ENCODE_COEFS_AC_FIRST(first_iter_ac_first_emit:, EMIT_ZRL_AC_FIRST,
                          EMIT_CODE_AC_FIRST);
#endif

    entropy->put_buffer = put_buffer;
    entropy->free_bits = free_bits;
    STORE_BUFFER()
  }

  if (cvalue < (values + Sl)) { /* If there are trailing zeroes, */
    entropy->EOBRUN++;          /* count an EOB */
    if (entropy->EOBRUN == 0x7FFF)
      emit_eobrun(entropy);     /* force it out to avoid overflow */
  }

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;

  /* Update restart-interval state too */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      entropy->restarts_to_go = cinfo->restart_interval;
      entropy->next_restart_num++;
      entropy->next_restart_num &= 7;
    }
    entropy->restarts_to_go--;
  }

  return TRUE;
}


/*
 * MCU encoding for DC successive approximation refinement scan.
 * Note: we assume such scans can be multi-component, although the spec
 * is not very clear on the point.
 */

METHODDEF(boolean)
encode_mcu_DC_refine(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  register int temp;
  int blkn;
  int Al = cinfo->Al;
  JBLOCKROW block;

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart(entropy, entropy->next_restart_num);

  /* Encode the MCU data blocks */
  for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
    block = MCU_data[blkn];

    /* We simply emit the Al'th bit of the DC coefficient value. */
    temp = (*block)[0];
    emit_bits(entropy, (unsigned int)(temp >> Al), 1);
  }

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;

  /* Update restart-interval state too */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      entropy->restarts_to_go = cinfo->restart_interval;
      entropy->next_restart_num++;
      entropy->next_restart_num &= 7;
    }
    entropy->restarts_to_go--;
  }

  return TRUE;
}


/*
 * Data preparation for encode_mcu_AC_refine().
 */

#define COMPUTE_ABSVALUES_AC_REFINE(Sl, koffset) { \
  /* It is convenient to make a pre-pass to determine the transformed \
   * coefficients' absolute values and the EOB position. \
   */ \
  for (k = 0; k < Sl; k++) { \
    temp = block[jpeg_natural_order_start[k]]; \
    /* We must apply the point transform by Al.  For AC coefficients this \
     * is an integer division with rounding towards 0.  To do this portably \
     * in C, we shift after obtaining the absolute value. \
     */ \
    temp2 = temp >> (CHAR_BIT * sizeof(int) - 1); \
    temp ^= temp2; \
    temp -= temp2;              /* temp is abs value of input */ \
    temp >>= Al;                /* apply the point transform */ \
    if (temp != 0) { \
      zerobits |= ((size_t)1U) << k; \
      signbits |= ((size_t)(temp2 + 1)) << k; \
    } \
    absvalues[k] = (UJCOEF)temp; /* save abs value for main pass */ \
    if (temp == 1) \
      EOB = k + koffset;        /* EOB = index of last newly-nonzero coef */ \
  } \
}

METHODDEF(int)
encode_mcu_AC_refine_prepare(const JCOEF *block,
                             const int *jpeg_natural_order_start, int Sl,
                             int Al, UJCOEF *absvalues, size_t *bits)
{
  register int k, temp, temp2;
  int EOB = 0;
  size_t zerobits = 0U, signbits = 0U;
  int Sl0 = Sl;

#if SIZEOF_SIZE_T == 4
  if (Sl0 > 32)
    Sl0 = 32;
#endif

  COMPUTE_ABSVALUES_AC_REFINE(Sl0, 0);

  bits[0] = zerobits;
#if SIZEOF_SIZE_T == 8
  bits[1] = signbits;
#else
  bits[2] = signbits;

  zerobits = 0U;
  signbits = 0U;

  if (Sl > 32) {
    Sl -= 32;
    jpeg_natural_order_start += 32;
    absvalues += 32;

    COMPUTE_ABSVALUES_AC_REFINE(Sl, 32);
  }

  bits[1] = zerobits;
  bits[3] = signbits;
#endif

  return EOB;
}


/*
 * MCU encoding for AC successive approximation refinement scan.
 */

/* The body of the per-coefficient encoding loop, parameterized on the
 * operations performed for a pending EOB run, a run-length-16 (ZRL) code, a
 * newly-nonzero coefficient code, and buffered correction bits, so that the
 * statistics-gathering and the data-output passes can each use a specialized
 * loop.
 */
#define ENCODE_COEFS_AC_REFINE(label, eobrun_op, zrl_op, code_op, corr_op) { \
  while (zerobits) { \
    idx = count_zeroes(&zerobits); \
    r += idx; \
    cabsvalue += idx; \
    signbits >>= idx; \
label \
    /* Emit any required ZRLs, but not if they can be folded into EOB */ \
    while (r > 15 && (cabsvalue <= EOBPTR)) { \
      /* emit any pending EOBRUN and the BE correction bits */ \
      eobrun_op \
      /* Emit ZRL */ \
      zrl_op \
      r -= 16; \
      /* Emit buffered correction bits that must be associated with ZRL */ \
      corr_op \
      BR_buffer = entropy->bit_buffer; /* BE bits are gone now */ \
      BR = 0; \
    } \
    \
    temp = *cabsvalue++; \
    \
    /* If the coef was previously nonzero, it only needs a correction bit. \
     * NOTE: a straight translation of the spec's figure G.7 would suggest \
     * that we also need to test r > 15.  But if r > 15, we can only get here \
     * if k > EOB, which implies that this coefficient is not 1. \
     */ \
    if (temp > 1) { \
      /* The correction bit is the next bit of the absolute value. */ \
      BR_buffer[BR++] = (char)(temp & 1); \
      signbits >>= 1; \
      zerobits >>= 1; \
      continue; \
    } \
    \
    /* Emit any pending EOBRUN and the BE correction bits */ \
    eobrun_op \
    \
    /* Count/emit Huffman symbol for run length / number of bits, along with \
     * the output bit for the newly-nonzero coef \
     */ \
    temp = signbits & 1; /* ((*block)[jpeg_natural_order_start[k]] < 0) ? 0 : 1 */ \
    code_op \
    \
    /* Emit buffered correction bits that must be associated with this code */ \
    corr_op \
    BR_buffer = entropy->bit_buffer; /* BE bits are gone now */ \
    BR = 0; \
    r = 0;                      /* reset zero run length */ \
    signbits >>= 1; \
    zerobits >>= 1; \
  } \
}

/* Count an EOB run/ZRL/coefficient symbol (statistics-gathering pass).
 * emit_eobrun() only updates the statistics counts in this mode, and
 * correction bits are not counted.
 */

#define COUNT_EOBRUN_AC_REFINE  { emit_eobrun(entropy); }

#define COUNT_ZRL_AC_REFINE     { ac_counts[0xF0]++; }

#define COUNT_CODE_AC_REFINE    { ac_counts[(r << 4) + 1]++; }

#define COUNT_CORR_AC_REFINE    { }

/* Emit an EOB run/ZRL/coefficient symbol/buffered correction bits
 * (data-output pass), inserting bits into the bit buffer held in local
 * variables.  These are functionally identical to emit_eobrun(),
 * emit_symbol(), emit_symbol_and_bits(), and emit_buffered_bits().
 */

#define PUT_BUFFERED_BITS(bufstart, count) { \
  const char *br_ptr = (bufstart); \
  unsigned int br_count = (count); \
  while (br_count >= 8) { \
    code = ((unsigned int)(br_ptr[0] & 1) << 7) | \
           ((unsigned int)(br_ptr[1] & 1) << 6) | \
           ((unsigned int)(br_ptr[2] & 1) << 5) | \
           ((unsigned int)(br_ptr[3] & 1) << 4) | \
           ((unsigned int)(br_ptr[4] & 1) << 3) | \
           ((unsigned int)(br_ptr[5] & 1) << 2) | \
           ((unsigned int)(br_ptr[6] & 1) << 1) | \
           ((unsigned int)(br_ptr[7] & 1)); \
    PUT_BITS(code, 8) \
    br_ptr += 8; \
    br_count -= 8; \
  } \
  while (br_count > 0) { \
    code = (unsigned int)(*br_ptr & 1); \
    PUT_BITS(code, 1) \
    br_ptr++; \
    br_count--; \
  } \
}

#define EMIT_EOBRUN_AC_REFINE { \
  if (entropy->EOBRUN > 0) { \
    unsigned int eobrun = entropy->EOBRUN; \
    \
    nbits = JPEG_NBITS_NONZERO(eobrun) - 1; \
    /* safety check: shouldn't happen given limited correction-bit buffer */ \
    if (nbits > 14) \
      ERREXIT(cinfo, JERR_HUFF_MISSING_CODE); \
    \
    symbol = nbits << 4; \
    size = actbl->ehufsi[symbol]; \
    if (size == 0) \
      ERREXIT(cinfo, JERR_HUFF_MISSING_CODE); \
    code = (actbl->ehufco[symbol] << nbits) | \
           (eobrun & ((1U << nbits) - 1)); \
    size += nbits; \
    PUT_BITS(code, size) \
    entropy->EOBRUN = 0; \
    \
    /* Emit any buffered correction bits */ \
    PUT_BUFFERED_BITS(entropy->bit_buffer, entropy->BE) \
    entropy->BE = 0; \
  } \
}

#define EMIT_ZRL_AC_REFINE { \
  if (zrl_size == 0) \
    ERREXIT(cinfo, JERR_HUFF_MISSING_CODE); \
  PUT_BITS(zrl_code, zrl_size) \
}

#define EMIT_CODE_AC_REFINE { \
  symbol = (r << 4) + 1; \
  size = actbl->ehufsi[symbol]; \
  /* if size is 0, caller used an invalid Huffman table entry */ \
  if (size == 0) \
    ERREXIT(cinfo, JERR_HUFF_MISSING_CODE); \
  code = (actbl->ehufco[symbol] << 1) | (unsigned int)temp; \
  size += 1; \
  PUT_BITS(code, size) \
}

#define EMIT_CORR_AC_REFINE  PUT_BUFFERED_BITS(BR_buffer, BR)

METHODDEF(boolean)
encode_mcu_AC_refine(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  register int temp, r, idx;
  char *BR_buffer;
  unsigned int BR;
  int Sl = cinfo->Se - cinfo->Ss + 1;
  int Al = cinfo->Al;
  UJCOEF absvalues_unaligned[DCTSIZE2 + 15];
  UJCOEF *absvalues;
  const UJCOEF *cabsvalue, *EOBPTR;
  size_t zerobits, signbits;
  size_t bits[16 / SIZEOF_SIZE_T];

#ifdef ZERO_BUFFERS
  memset(absvalues_unaligned, 0, sizeof(absvalues_unaligned));
  memset(bits, 0, sizeof(bits));
#endif

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart(entropy, entropy->next_restart_num);

#ifdef WITH_SIMD
  cabsvalue = absvalues = (UJCOEF *)PAD((JUINTPTR)absvalues_unaligned, 16);
#else
  /* Not using SIMD, so alignment is not needed */
  cabsvalue = absvalues = absvalues_unaligned;
#endif

  /* Prepare data */
  EOBPTR = absvalues +
    entropy->AC_refine_prepare(MCU_data[0][0], jpeg_natural_order + cinfo->Ss,
                               Sl, Al, absvalues, bits);

  /* Encode the AC coefficients per section G.1.2.3, fig. G.7 */

  r = 0;                        /* r = run length of zeros */
  BR = 0;                       /* BR = count of buffered bits added now */
  BR_buffer = entropy->bit_buffer + entropy->BE; /* Append bits to buffer */

  zerobits = bits[0];
#if SIZEOF_SIZE_T == 8
  signbits = bits[1];
#else
  signbits = bits[2];
#endif

  if (entropy->gather_statistics) {
    long *ac_counts = entropy->count_ptrs[entropy->ac_tbl_no];

    ENCODE_COEFS_AC_REFINE((void)0;, COUNT_EOBRUN_AC_REFINE,
                           COUNT_ZRL_AC_REFINE, COUNT_CODE_AC_REFINE,
                           COUNT_CORR_AC_REFINE);

#if SIZEOF_SIZE_T == 4
    zerobits = bits[1];
    signbits = bits[3];

    if (zerobits) {
      int diff = ((absvalues + DCTSIZE2 / 2) - cabsvalue);
      idx = count_zeroes(&zerobits);
      signbits >>= idx;
      idx += diff;
      r += idx;
      cabsvalue += idx;
      goto first_iter_ac_refine_gather;
    }

    ENCODE_COEFS_AC_REFINE(first_iter_ac_refine_gather:,
                           COUNT_EOBRUN_AC_REFINE, COUNT_ZRL_AC_REFINE,
                           COUNT_CODE_AC_REFINE, COUNT_CORR_AC_REFINE);
#endif
  } else {
    c_derived_tbl *actbl = entropy->derived_tbls[entropy->ac_tbl_no];
    unsigned int zrl_code = actbl->ehufco[0xF0];
    int zrl_size = actbl->ehufsi[0xF0];
    unsigned int code;
    int symbol, size, nbits;
    JOCTET _buffer[BUFSIZE_AC_REFINE], *buffer;
    bit_buf_type put_buffer = entropy->put_buffer;
    int free_bits = entropy->free_bits;
    int localbuf = 0;

#ifdef ZERO_BUFFERS
    memset(_buffer, 0, sizeof(_buffer));
#endif

    LOAD_BUFFER(BUFSIZE_AC_REFINE)

    ENCODE_COEFS_AC_REFINE((void)0;, EMIT_EOBRUN_AC_REFINE,
                           EMIT_ZRL_AC_REFINE, EMIT_CODE_AC_REFINE,
                           EMIT_CORR_AC_REFINE);

#if SIZEOF_SIZE_T == 4
    zerobits = bits[1];
    signbits = bits[3];

    if (zerobits) {
      int diff = ((absvalues + DCTSIZE2 / 2) - cabsvalue);
      idx = count_zeroes(&zerobits);
      signbits >>= idx;
      idx += diff;
      r += idx;
      cabsvalue += idx;
      goto first_iter_ac_refine_emit;
    }

    ENCODE_COEFS_AC_REFINE(first_iter_ac_refine_emit:,
                           EMIT_EOBRUN_AC_REFINE, EMIT_ZRL_AC_REFINE,
                           EMIT_CODE_AC_REFINE, EMIT_CORR_AC_REFINE);
#endif

    entropy->put_buffer = put_buffer;
    entropy->free_bits = free_bits;
    STORE_BUFFER()
  }

  r |= (int)((absvalues + Sl) - cabsvalue);

  if (r > 0 || BR > 0) {        /* If there are trailing zeroes, */
    entropy->EOBRUN++;          /* count an EOB */
    entropy->BE += BR;          /* concat my correction bits to older ones */
    /* We force out the EOB if we risk either:
     * 1. overflow of the EOB counter;
     * 2. overflow of the correction bit buffer during the next MCU.
     */
    if (entropy->EOBRUN == 0x7FFF ||
        entropy->BE > (MAX_CORR_BITS - DCTSIZE2 + 1))
      emit_eobrun(entropy);
  }

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;

  /* Update restart-interval state too */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      entropy->restarts_to_go = cinfo->restart_interval;
      entropy->next_restart_num++;
      entropy->next_restart_num &= 7;
    }
    entropy->restarts_to_go--;
  }

  return TRUE;
}


/*
 * Finish up at the end of a Huffman-compressed progressive scan.
 */

METHODDEF(void)
finish_pass_phuff(j_compress_ptr cinfo)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Flush out any buffered data */
  emit_eobrun(entropy);
  flush_bits(entropy);

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;
}


/*
 * Finish up a statistics-gathering pass and create the new Huffman tables.
 */

METHODDEF(void)
finish_pass_gather_phuff(j_compress_ptr cinfo)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  boolean is_DC_band;
  int ci, tbl;
  jpeg_component_info *compptr;
  JHUFF_TBL **htblptr;
  boolean did[NUM_HUFF_TBLS];
  long counts[257];
  unsigned long total_bits = 0;

  /* Flush out buffered data (all we care about is counting the EOB symbol) */
  emit_eobrun(entropy);

  is_DC_band = (cinfo->Ss == 0);

  /* It's important not to apply jpeg_gen_optimal_table more than once
   * per table, because it clobbers the input frequency counts!
   */
  memset(did, 0, sizeof(did));

  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    if (is_DC_band) {
      if (cinfo->Ah != 0)       /* DC refinement needs no table */
        continue;
      tbl = compptr->dc_tbl_no;
    } else {
      tbl = compptr->ac_tbl_no;
    }
    if (!did[tbl]) {
      if (is_DC_band)
        htblptr = &cinfo->dc_huff_tbl_ptrs[tbl];
      else
        htblptr = &cinfo->ac_huff_tbl_ptrs[tbl];
      if (*htblptr == NULL)
        *htblptr = jpeg_alloc_huff_table((j_common_ptr)cinfo);
      memcpy(counts, entropy->count_ptrs[tbl], sizeof(counts));
      jpeg_gen_optimal_table(cinfo, *htblptr, entropy->count_ptrs[tbl]);
      did[tbl] = TRUE;

      /* Compute the exact number of bits with which the symbols counted
       * during this pass, along with their appended magnitude/EOB-run bits,
       * will be entropy-coded using the Huffman table that was just
       * generated.  This is a lower bound on the encoded size of the scan
       * (markers, byte stuffing, and -- in AC refinement scans -- correction
       * bits only add to it), which jcmaster.c uses to skip data-output
       * passes whose result provably cannot affect scan selection.
       */
      {
        JHUFF_TBL *htbl = *htblptr;
        int bl, j, idx = 0;

        for (bl = 1; bl <= 16; bl++) {
          for (j = 0; j < htbl->bits[bl]; j++) {
            int sym = htbl->huffval[idx++];
            int extra;

            if (is_DC_band)
              extra = sym;            /* DC difference magnitude bits */
            else if (sym & 15)
              extra = sym & 15;       /* AC coefficient magnitude bits */
            else if ((sym >> 4) == 15)
              extra = 0;              /* ZRL */
            else
              extra = sym >> 4;       /* EOB run length bits */
            total_bits +=
              (unsigned long)counts[sym] * (unsigned long)(bl + extra);
          }
        }
      }
    }
  }

  cinfo->master->scan_size_lower_bound = total_bits / 8;
}


/*
 * Module initialization routine for progressive Huffman entropy encoding.
 */

GLOBAL(void)
jinit_phuff_encoder(j_compress_ptr cinfo)
{
  phuff_entropy_ptr entropy;
  int i;

  entropy = (phuff_entropy_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(phuff_entropy_encoder));
  cinfo->entropy = (struct jpeg_entropy_encoder *)entropy;
  entropy->pub.start_pass = start_pass_phuff;

  /* Mark tables unallocated */
  for (i = 0; i < NUM_HUFF_TBLS; i++) {
    entropy->derived_tbls[i] = NULL;
    entropy->count_ptrs[i] = NULL;
  }
  entropy->bit_buffer = NULL;   /* needed only in AC refinement scan */
}

#endif /* C_PROGRESSIVE_SUPPORTED */

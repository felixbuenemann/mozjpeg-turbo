/*
 * jcparopt.c
 *
 * Copyright (C) 2026, Mozilla Corporation.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains an optional, multithreaded implementation of the
 * candidate-scan evaluation that progressive scan optimization
 * (cinfo->master->optimize_scans) performs.
 *
 * When scan optimization is enabled, most of the compression time is spent
 * encoding candidate scans whose sizes are compared in select_scans()
 * (jcmaster.c); the encoded buffers of the winning candidates are
 * concatenated to form the output image.  Every candidate scan is encoded
 * from the same, final coefficient buffer, and its encoded bytes depend only
 * on its own scan parameters and coefficients, so the candidates can be
 * encoded concurrently.
 *
 * jpeg_par_scan_opt_run() encodes the candidate scans on worker threads
 * ahead of time, each worker using a private jpeg_compress_struct (and
 * therefore a private memory pool, entropy encoder, marker writer, and
 * memory destination) that reads the shared coefficient buffer.  The
 * regular, sequential pass machinery in jcmaster.c then "replays" the
 * precomputed buffers: the statistics-gathering and data-output passes of a
 * precomputed scan become no-ops, and the precomputed buffer and size are
 * installed when the scan's output pass completes, after which scan
 * selection proceeds unchanged.  The compressed output is therefore
 * bit-for-bit identical to that of a single-threaded build.  A scan whose
 * precomputation failed (or that was never precomputed) is simply encoded
 * by the regular machinery.
 *
 * The searches in select_scans() do not evaluate every candidate scan: the
 * successive-approximation searches stop at the first bit precision that
 * does not improve on the best cost, and the frequency-split searches have
 * early-termination heuristics.  To avoid spending CPU time on candidates
 * that the searches would never evaluate, the workers do not simply encode
 * everything: a small scheduler replicates the decision structure of
 * select_scans(), releases jobs in search order (plus a configurable amount
 * of speculation), evaluates each search decision as soon as the sizes it
 * depends on are available, and cancels the jobs that the search outcome
 * proves unreachable.  In addition, after a candidate scan's
 * statistics-gathering pass, its data-output pass is skipped if the scan's
 * size lower bound (see finish_pass_gather_phuff() in jcphuff.c) already
 * proves that its comparison must fail; the lower bound is then recorded as
 * the scan's size, exactly as the sequential pruning in jcmaster.c would.
 * (This proof remains valid under speculation because the best cost of a
 * search only decreases as it progresses.)
 *
 * The frequency-split candidate scans use the Al value chosen by the
 * successive-approximation searches, so their jobs only become eligible
 * once the corresponding search has been resolved; the Al value used is
 * re-verified at replay time, and the precomputed result is discarded on a
 * mismatch.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jcmaster.h"
#include "jchuff.h"
#include "jpeg_nbits.h"

#ifdef WITH_SCAN_OPT_THREADS

#include <pthread.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PAR_MAX_THREADS  16

/* Number of search steps (successive-approximation precisions or frequency
 * splits) that may be encoded speculatively beyond the step the search is
 * currently deciding.  Speculation trades CPU time for better thread
 * utilization around search decision points: the default of 2 is within a
 * few percent of the best achievable wall-clock time at roughly 25-35% CPU
 * overhead, while 0 evaluates exactly the candidates the sequential search
 * would, making the total CPU time nearly identical to that of a
 * sequential build.  Overridden by the MOZJPEG_SCAN_SPECULATION environment
 * variable.
 */
#define PAR_DEFAULT_SPECULATION  2

/* Job states */
#define JOB_BLOCKED    0        /* not yet released by the scheduler */
#define JOB_READY      1        /* may be picked up by a worker */
#define JOB_RUNNING    2
#define JOB_DONE       3        /* finished (successfully or not) */
#define JOB_CANCELLED  4        /* provably not evaluated by the search */

/* Per-scan result states (par_state::valid) */
#define RESULT_NONE    0
#define RESULT_BUFFER  1        /* encoded buffer available */
#define RESULT_BOUND   2        /* provably-losing scan: size lower bound
                                   recorded, no buffer (as with the
                                   sequential pruning in jcmaster.c) */

/* State of one search (successive-approximation or frequency-split) */
typedef struct {
  int decided;                  /* # of steps whose comparison is resolved */
  int released;                 /* upper end of the released window */
  boolean best_set;             /* best is valid */
  boolean stopped;              /* search terminated early */
  boolean resolved;             /* outcome (best_idx) is final */
  unsigned long best;
  int best_idx;
} par_chain;

typedef struct {
  int num_scans;
  unsigned char **buf;          /* per-scan encoded buffer (malloc) */
  unsigned long *size;          /* per-scan encoded size (or lower bound) */
  int *valid;                   /* per-scan RESULT_* state */
  int *al;                      /* per-scan: Al value used for encoding */

  /* Read-only block row pointers into the shared coefficient buffer */
  JBLOCKROW *rows[MAX_COMPONENTS];

  j_compress_ptr maincinfo;

  /* Scheduler state, guarded by lock */
  pthread_mutex_t lock;
  pthread_cond_t cond;
  int *job_state;
  int *job_al;                  /* Al value to encode the scan with */
  int num_terminal;             /* # of jobs in a terminal state */
  int speculation;
  boolean abandoned;            /* a worker failed: stop making decisions */

  /* Scan script geometry (see select_scans() in jcmaster.c) */
  int lfss, nsl, base, cfss;
  int almax_luma, almax_chroma, nfs;

  par_chain luma_al, luma_fs, chroma_al, chroma_fs;
} par_state;

typedef struct {
  struct jpeg_error_mgr pub;
  jmp_buf jb;
} par_error_mgr;


static void
par_error_exit(j_common_ptr cinfo)
{
  longjmp(((par_error_mgr *)cinfo->err)->jb, 1);
}

static void
par_output_message(j_common_ptr cinfo)
{
  /* Suppress worker messages; a failed scan is re-encoded sequentially,
   * which reports the error through the application's error manager.
   */
}


/*
 * Scheduler helpers.  All of these must be called with par->lock held.
 */

static void
par_release_job(par_state *par, int n, int Al)
{
  if (n < par->num_scans && par->job_state[n] == JOB_BLOCKED) {
    par->job_state[n] = JOB_READY;
    par->job_al[n] = Al;
  }
}

static void
par_cancel_job(par_state *par, int n)
{
  /* Jobs that have not been picked up by a worker yet (whether or not they
   * have been released, e.g. speculatively) can be cancelled.
   */
  if (n < par->num_scans && (par->job_state[n] == JOB_BLOCKED ||
                             par->job_state[n] == JOB_READY)) {
    par->job_state[n] = JOB_CANCELLED;
    par->num_terminal++;
  }
}

static boolean
par_size_known(par_state *par, int n)
{
  return par->valid[n] != RESULT_NONE;
}

/* Scan index helpers for the four searches */

#define LUMA_AL_MEMBER(par, level, m)    (3 * (level) + 1 + (m))   /* m=0,1 */
#define LUMA_AL_REFINE(par, level)       (3 * (level) + 3)
#define LUMA_FS_NOSPLIT(par)             ((par)->lfss)
#define LUMA_FS_PAIR(par, k, m)          ((par)->lfss + 2 * (k) - 1 + (m))
#define CHROMA_AL_MEMBER(par, level, m)  ((par)->base + 6 * (level) + (m))
#define CHROMA_AL_REFINE(par, level, m)  ((par)->base + 6 * (level) + 4 + (m))
#define CHROMA_FS_NOSPLIT(par, m)        ((par)->cfss + (m))
#define CHROMA_FS_QUAD(par, k, m)        ((par)->cfss + 4 * (k) - 2 + (m))

/*
 * Advance the searches: resolve any decisions whose input sizes are
 * available, release newly-eligible jobs (within the speculation window),
 * and cancel jobs that the search outcomes prove unreachable.  This
 * replicates the comparison logic of select_scans() (jcmaster.c) exactly,
 * using the same sizes that will be installed for the sequential replay.
 */

static void
par_advance(par_state *par)
{
  j_compress_ptr cinfo = par->maincinfo;
  par_chain *ch;
  unsigned long cost;
  int level, k, i, m, limit;

  if (par->abandoned) {
    /* A worker failed; the sequential machinery will encode whatever is
     * missing.  Cancel all unreleased jobs so that the workers terminate.
     */
    for (i = 0; i < par->num_scans; i++)
      par_cancel_job(par, i);
    return;
  }

  /* Luma successive-approximation search */
  ch = &par->luma_al;
  while (!ch->stopped && ch->decided <= par->almax_luma) {
    level = ch->decided;
    if (!par_size_known(par, LUMA_AL_MEMBER(par, level, 0)) ||
        !par_size_known(par, LUMA_AL_MEMBER(par, level, 1)))
      break;
    cost = par->size[LUMA_AL_MEMBER(par, level, 0)] +
           par->size[LUMA_AL_MEMBER(par, level, 1)];
    for (i = 0; i < level; i++) {
      if (!par_size_known(par, LUMA_AL_REFINE(par, i)))
        break;
      cost += par->size[LUMA_AL_REFINE(par, i)];
    }
    if (i < level)
      break;
    ch->decided++;
    /* NOTE: the last precision has no refinement scan (its index would be
     * that of the unsplit frequency-split scan.)
     */
    if (level == 0 || cost < ch->best) {
      ch->best = cost;
      ch->best_set = TRUE;
      ch->best_idx = level;
      if (level < par->almax_luma)
        par_release_job(par, LUMA_AL_REFINE(par, level),
                        cinfo->scan_info[LUMA_AL_REFINE(par, level)].Al);
    } else {
      ch->stopped = TRUE;
      if (level < par->almax_luma)
        par_cancel_job(par, LUMA_AL_REFINE(par, level));
      for (i = level + 1; i <= par->almax_luma; i++) {
        par_cancel_job(par, LUMA_AL_MEMBER(par, i, 0));
        par_cancel_job(par, LUMA_AL_MEMBER(par, i, 1));
        if (i < par->almax_luma)
          par_cancel_job(par, LUMA_AL_REFINE(par, i));
      }
    }
  }
  if (ch->stopped || ch->decided > par->almax_luma)
    ch->resolved = TRUE;

  /* Release luma SA member jobs within the speculation window */
  if (!ch->stopped) {
    limit = ch->decided + par->speculation;
    if (limit > par->almax_luma)
      limit = par->almax_luma;
    for (level = ch->released; level <= limit; level++) {
      for (m = 0; m < 2; m++) {
        i = LUMA_AL_MEMBER(par, level, m);
        par_release_job(par, i, cinfo->scan_info[i].Al);
      }
      /* The refinement scan is only encoded by the sequential search if
       * the precision improves on the best cost, so without speculation it
       * is released by the decision above instead.
       */
      if (level < ch->decided + par->speculation &&
          level < par->almax_luma) {
        i = LUMA_AL_REFINE(par, level);
        par_release_job(par, i, cinfo->scan_info[i].Al);
      }
    }
    if (ch->released <= limit)
      ch->released = limit + 1;
  }

  /* Luma frequency-split search (eligible once the SA search is resolved,
   * which determines the Al value of its candidate scans)
   */
  if (par->luma_al.resolved && par->nsl > par->lfss) {
    int Al = par->luma_al.best_idx;

    ch = &par->luma_fs;
    par_release_job(par, LUMA_FS_NOSPLIT(par), Al);
    if (!ch->best_set && par_size_known(par, LUMA_FS_NOSPLIT(par))) {
      ch->best = par->size[LUMA_FS_NOSPLIT(par)];
      ch->best_set = TRUE;
      ch->best_idx = 0;
    }
    while (ch->best_set && !ch->stopped && ch->decided < par->nfs) {
      k = ch->decided + 1;
      if (!par_size_known(par, LUMA_FS_PAIR(par, k, 0)) ||
          !par_size_known(par, LUMA_FS_PAIR(par, k, 1)))
        break;
      cost = par->size[LUMA_FS_PAIR(par, k, 0)] +
             par->size[LUMA_FS_PAIR(par, k, 1)];
      ch->decided++;
      if (cost < ch->best) {
        ch->best = cost;
        ch->best_idx = k;
      }
      /* if after testing first 3, no split is the best, don't search
       * further (see select_scans())
       */
      if ((k == 2 && ch->best_idx == 0) || (k == 3 && ch->best_idx != 2) ||
          (k == 4 && ch->best_idx != 4)) {
        ch->stopped = TRUE;
        for (i = k + 1; i <= par->nfs; i++) {
          par_cancel_job(par, LUMA_FS_PAIR(par, i, 0));
          par_cancel_job(par, LUMA_FS_PAIR(par, i, 1));
        }
      }
    }
    if (!ch->stopped && ch->best_set) {
      limit = ch->decided + 1 + par->speculation;
      if (limit > par->nfs)
        limit = par->nfs;
      for (k = ch->released + 1; k <= limit; k++) {
        par_release_job(par, LUMA_FS_PAIR(par, k, 0), Al);
        par_release_job(par, LUMA_FS_PAIR(par, k, 1), Al);
      }
      if (ch->released < limit)
        ch->released = limit;
    }
  }

  if (par->num_scans <= par->nsl)       /* no chroma (grayscale) */
    return;

  /* Chroma successive-approximation search */
  ch = &par->chroma_al;
  while (!ch->stopped && ch->decided <= par->almax_chroma) {
    level = ch->decided;
    cost = 0;
    for (m = 0; m < 4; m++) {
      if (!par_size_known(par, CHROMA_AL_MEMBER(par, level, m)))
        break;
      cost += par->size[CHROMA_AL_MEMBER(par, level, m)];
    }
    if (m < 4)
      break;
    for (i = 0; i < level; i++) {
      if (!par_size_known(par, CHROMA_AL_REFINE(par, i, 0)) ||
          !par_size_known(par, CHROMA_AL_REFINE(par, i, 1)))
        break;
      cost += par->size[CHROMA_AL_REFINE(par, i, 0)];
      cost += par->size[CHROMA_AL_REFINE(par, i, 1)];
    }
    if (i < level)
      break;
    ch->decided++;
    /* NOTE: the last precision has no refinement scans (their indices would
     * be those of the unsplit chroma frequency-split scans.)
     */
    if (level == 0 || cost < ch->best) {
      ch->best = cost;
      ch->best_set = TRUE;
      ch->best_idx = level;
      if (level < par->almax_chroma) {
        for (m = 0; m < 2; m++) {
          i = CHROMA_AL_REFINE(par, level, m);
          par_release_job(par, i, cinfo->scan_info[i].Al);
        }
      }
    } else {
      ch->stopped = TRUE;
      if (level < par->almax_chroma) {
        for (m = 0; m < 2; m++)
          par_cancel_job(par, CHROMA_AL_REFINE(par, level, m));
      }
      for (i = level + 1; i <= par->almax_chroma; i++) {
        for (m = 0; m < 4; m++)
          par_cancel_job(par, CHROMA_AL_MEMBER(par, i, m));
        if (i < par->almax_chroma)
          for (m = 0; m < 2; m++)
            par_cancel_job(par, CHROMA_AL_REFINE(par, i, m));
      }
    }
  }
  if (ch->stopped || ch->decided > par->almax_chroma)
    ch->resolved = TRUE;

  if (!ch->stopped) {
    limit = ch->decided + par->speculation;
    if (limit > par->almax_chroma)
      limit = par->almax_chroma;
    for (level = ch->released; level <= limit; level++) {
      for (m = 0; m < 4; m++) {
        i = CHROMA_AL_MEMBER(par, level, m);
        par_release_job(par, i, cinfo->scan_info[i].Al);
      }
      if (level < ch->decided + par->speculation &&
          level < par->almax_chroma) {
        for (m = 0; m < 2; m++) {
          i = CHROMA_AL_REFINE(par, level, m);
          par_release_job(par, i, cinfo->scan_info[i].Al);
        }
      }
    }
    if (ch->released <= limit)
      ch->released = limit + 1;
  }

  /* Chroma frequency-split search */
  if (par->chroma_al.resolved && par->num_scans > par->cfss) {
    int Al = par->chroma_al.best_idx;

    ch = &par->chroma_fs;
    par_release_job(par, CHROMA_FS_NOSPLIT(par, 0), Al);
    par_release_job(par, CHROMA_FS_NOSPLIT(par, 1), Al);
    if (!ch->best_set && par_size_known(par, CHROMA_FS_NOSPLIT(par, 0)) &&
        par_size_known(par, CHROMA_FS_NOSPLIT(par, 1))) {
      ch->best = par->size[CHROMA_FS_NOSPLIT(par, 0)] +
                 par->size[CHROMA_FS_NOSPLIT(par, 1)];
      ch->best_set = TRUE;
      ch->best_idx = 0;
    }
    while (ch->best_set && !ch->stopped && ch->decided < par->nfs) {
      k = ch->decided + 1;
      cost = 0;
      for (m = 0; m < 4; m++) {
        if (!par_size_known(par, CHROMA_FS_QUAD(par, k, m)))
          break;
        cost += par->size[CHROMA_FS_QUAD(par, k, m)];
      }
      if (m < 4)
        break;
      ch->decided++;
      if (cost < ch->best) {
        ch->best = cost;
        ch->best_idx = k;
      }
      if ((k == 2 && ch->best_idx == 0) || (k == 3 && ch->best_idx != 2) ||
          (k == 4 && ch->best_idx != 4)) {
        ch->stopped = TRUE;
        for (i = k + 1; i <= par->nfs; i++)
          for (m = 0; m < 4; m++)
            par_cancel_job(par, CHROMA_FS_QUAD(par, i, m));
      }
    }
    if (!ch->stopped && ch->best_set) {
      limit = ch->decided + 1 + par->speculation;
      if (limit > par->nfs)
        limit = par->nfs;
      for (k = ch->released + 1; k <= limit; k++)
        for (m = 0; m < 4; m++)
          par_release_job(par, CHROMA_FS_QUAD(par, k, m), Al);
      if (ch->released < limit)
        ch->released = limit;
    }
  }
}


/*
 * Can the data-output pass of the given scan be skipped?  Mirrors the
 * reasoning of scan_emit_provably_useless() (jcmaster.c): if the comparison
 * the scan participates in must fail even when its own size and any unknown
 * sizes are replaced with lower bounds (or zero), then the scan can never
 * be selected, and the lower bound can be recorded as its size.  Scans that
 * can be selected for output are never skipped.  The proof remains valid
 * under speculation because a search's best cost only decreases.  Called
 * with par->lock held.
 */

static boolean
par_emit_provably_useless(par_state *par, int n, unsigned long lb)
{
  unsigned long cost;
  int level, k, i, m;

  if (lb == 0 || n >= par->num_scans - 1)
    return FALSE;

  if (n >= 1 && n < par->lfss) {
    /* luma SA member (refinement scans are never skipped) */
    m = (n - 1) % 3;
    level = (n - 1) / 3;
    if (m == 2 || level == 0 || !par->luma_al.best_set)
      return FALSE;
    cost = lb;
    for (i = 0; i < 2; i++) {
      k = LUMA_AL_MEMBER(par, level, i);
      if (k != n && par_size_known(par, k))
        cost += par->size[k];
    }
    for (i = 0; i < level; i++)
      if (par_size_known(par, LUMA_AL_REFINE(par, i)))
        cost += par->size[LUMA_AL_REFINE(par, i)];
    return cost >= par->luma_al.best;

  } else if (n > par->lfss && n < par->nsl) {
    if (!par->luma_fs.best_set)
      return FALSE;
    k = (n - par->lfss + 1) / 2;
    cost = lb;
    for (i = 0; i < 2; i++) {
      m = LUMA_FS_PAIR(par, k, i);
      if (m != n && par_size_known(par, m))
        cost += par->size[m];
    }
    return cost >= par->luma_fs.best;

  } else if (n >= par->base && n < par->cfss) {
    m = (n - par->base) % 6;
    level = (n - par->base) / 6;
    if (m >= 4 || level == 0 || !par->chroma_al.best_set)
      return FALSE;
    cost = lb;
    for (i = 0; i < 4; i++) {
      k = CHROMA_AL_MEMBER(par, level, i);
      if (k != n && par_size_known(par, k))
        cost += par->size[k];
    }
    for (i = 0; i < level; i++) {
      if (par_size_known(par, CHROMA_AL_REFINE(par, i, 0)))
        cost += par->size[CHROMA_AL_REFINE(par, i, 0)];
      if (par_size_known(par, CHROMA_AL_REFINE(par, i, 1)))
        cost += par->size[CHROMA_AL_REFINE(par, i, 1)];
    }
    return cost >= par->chroma_al.best;

  } else if (n > par->cfss + 1 && n < par->num_scans) {
    if (!par->chroma_fs.best_set)
      return FALSE;
    k = (n - par->cfss + 2) / 4;
    cost = lb;
    for (i = 0; i < 4; i++) {
      m = CHROMA_FS_QUAD(par, k, i);
      if (m != n && par_size_known(par, m))
        cost += par->size[m];
    }
    return cost >= par->chroma_fs.best;
  }

  return FALSE;
}


/*
 * Record a provably-losing scan's lower bound in place of its size and
 * skip its data-output pass.  Called by a worker between the two passes of
 * a scan.
 */

static boolean
par_try_skip_emit(par_state *par, int n, int Al, unsigned long lb)
{
  boolean skip;

  pthread_mutex_lock(&par->lock);
  skip = par_emit_provably_useless(par, n, lb);
  if (skip) {
    par->size[n] = lb;
    par->al[n] = Al;
    par->valid[n] = RESULT_BOUND;
  }
  pthread_mutex_unlock(&par->lock);
  return skip;
}


/*
 * Feed one scan's worth of MCUs from the shared coefficient buffer to the
 * worker's entropy encoder.  This replicates the iteration that
 * compress_output() (jccoefct.c) performs in JBUF_CRANK_DEST mode.
 */

static void
par_iterate_MCUs(par_state *par, j_compress_ptr wc)
{
  JBLOCKROW MCU_buffer[C_MAX_BLOCKS_IN_MCU];
  JDIMENSION iMCU_row, MCU_col_num, start_col, blk_row;
  int blkn, ci, xindex, yindex, yoffset, MCU_rows;
  jpeg_component_info *compptr;

  for (iMCU_row = 0; iMCU_row < wc->total_iMCU_rows; iMCU_row++) {
    /* In an interleaved scan, an MCU row is the same as an iMCU row.
     * In a noninterleaved scan, an iMCU row has v_samp_factor MCU rows.
     * But at the bottom of the image, process only what's left.
     */
    if (wc->comps_in_scan > 1)
      MCU_rows = 1;
    else if (iMCU_row < wc->total_iMCU_rows - 1)
      MCU_rows = wc->cur_comp_info[0]->v_samp_factor;
    else
      MCU_rows = wc->cur_comp_info[0]->last_row_height;

    for (yoffset = 0; yoffset < MCU_rows; yoffset++) {
      for (MCU_col_num = 0; MCU_col_num < wc->MCUs_per_row; MCU_col_num++) {
        /* Construct list of pointers to DCT blocks belonging to this MCU */
        blkn = 0;
        for (ci = 0; ci < wc->comps_in_scan; ci++) {
          compptr = wc->cur_comp_info[ci];
          start_col = MCU_col_num * compptr->MCU_width;
          blk_row = iMCU_row * compptr->v_samp_factor + yoffset;
          for (yindex = 0; yindex < compptr->MCU_height; yindex++) {
            JBLOCKROW buffer_ptr =
              par->rows[compptr->component_index][blk_row + yindex] +
              start_col;
            for (xindex = 0; xindex < compptr->MCU_width; xindex++)
              MCU_buffer[blkn++] = buffer_ptr++;
          }
        }
        if (!(*wc->entropy->encode_mcu) (wc, MCU_buffer))
          ERREXIT(wc, JERR_CANT_SUSPEND);
      }
    }
  }
}


/*
 * When Huffman tables are not optimized, each table is emitted by the first
 * scan that uses it, and the marker writer's sent_table state spans scans.
 * Determine whether the given table has been emitted by a scan before the
 * given one, replicating the emission conditions of write_scan_header()
 * (jcmarker.c.)
 */

static boolean
par_huff_table_sent(j_compress_ptr maininfo, int scan_number, int tblno,
                    boolean is_dc)
{
  int m, ci;

  for (m = 0; m < scan_number; m++) {
    const jpeg_scan_info *sp = maininfo->scan_info + m;

    for (ci = 0; ci < sp->comps_in_scan; ci++) {
      jpeg_component_info *compptr =
        &maininfo->comp_info[sp->component_index[ci]];

      if (is_dc) {
        if (sp->Ss == 0 && sp->Ah == 0 && compptr->dc_tbl_no == tblno)
          return TRUE;
      } else {
        if (sp->Se > 0 && compptr->ac_tbl_no == tblno)
          return TRUE;
      }
    }
  }
  return FALSE;
}


/*
 * Encode one candidate scan into a malloc'd buffer, using a private
 * compression object.  Returns RESULT_NONE on failure, RESULT_BUFFER on
 * success, or RESULT_BOUND if the data-output pass was skipped because the
 * scan provably cannot be selected.
 */

static int
par_encode_scan(par_state *par, int scan_number, int Al)
{
  j_compress_ptr maininfo = par->maincinfo;
  const jpeg_scan_info *scanptr = maininfo->scan_info + scan_number;
  struct jpeg_compress_struct wc;
  par_error_mgr werr;
  unsigned char *buf = NULL;
  unsigned long bufsize = 0;
  unsigned int prev_restart_interval = 0, saved_restart_interval;
  int ci, i;

  memset(&wc, 0, sizeof(wc));
  wc.err = jpeg_std_error(&werr.pub);
  werr.pub.error_exit = par_error_exit;
  werr.pub.output_message = par_output_message;

  if (setjmp(werr.jb)) {
    /* The worker's error manager longjmps here on any error. */
    jpeg_destroy_compress(&wc);
    free(buf);
    return RESULT_NONE;
  }

  jpeg_create_compress(&wc);

  /* Copy the parameters that the coefficient iteration, the entropy
   * encoder, and the marker writer depend on.
   */
  wc.image_width = maininfo->image_width;
  wc.image_height = maininfo->image_height;
  wc.data_precision = maininfo->data_precision;
  wc.num_components = maininfo->num_components;
  wc.jpeg_color_space = maininfo->jpeg_color_space;
  wc.max_h_samp_factor = maininfo->max_h_samp_factor;
  wc.max_v_samp_factor = maininfo->max_v_samp_factor;
  wc.total_iMCU_rows = maininfo->total_iMCU_rows;
  wc.progressive_mode = TRUE;
  wc.optimize_coding = maininfo->optimize_coding;
  wc.arith_code = maininfo->arith_code;
  wc.restart_interval = maininfo->restart_interval;
  wc.restart_in_rows = maininfo->restart_in_rows;
  memcpy(wc.arith_dc_L, maininfo->arith_dc_L, sizeof(wc.arith_dc_L));
  memcpy(wc.arith_dc_U, maininfo->arith_dc_U, sizeof(wc.arith_dc_U));
  memcpy(wc.arith_ac_K, maininfo->arith_ac_K, sizeof(wc.arith_ac_K));

  wc.comp_info = (jpeg_component_info *)
    (*wc.mem->alloc_small) ((j_common_ptr)&wc, JPOOL_IMAGE,
                            wc.num_components * sizeof(jpeg_component_info));
  memcpy(wc.comp_info, maininfo->comp_info,
         wc.num_components * sizeof(jpeg_component_info));

  for (i = 0; i < NUM_QUANT_TBLS; i++) {
    if (maininfo->quant_tbl_ptrs[i] != NULL) {
      wc.quant_tbl_ptrs[i] = jpeg_alloc_quant_table((j_common_ptr)&wc);
      memcpy(wc.quant_tbl_ptrs[i]->quantval,
             maininfo->quant_tbl_ptrs[i]->quantval,
             sizeof(wc.quant_tbl_ptrs[i]->quantval));
      wc.quant_tbl_ptrs[i]->sent_table = FALSE;
    }
  }

  if (!wc.arith_code && !wc.optimize_coding) {
    /* Huffman compression with fixed tables: copy the tables, replicating
     * the marker writer's cross-scan sent_table state.
     */
    for (i = 0; i < NUM_HUFF_TBLS; i++) {
      if (maininfo->dc_huff_tbl_ptrs[i] != NULL) {
        wc.dc_huff_tbl_ptrs[i] = jpeg_alloc_huff_table((j_common_ptr)&wc);
        memcpy(wc.dc_huff_tbl_ptrs[i], maininfo->dc_huff_tbl_ptrs[i],
               sizeof(JHUFF_TBL));
        wc.dc_huff_tbl_ptrs[i]->sent_table =
          par_huff_table_sent(maininfo, scan_number, i, TRUE);
      }
      if (maininfo->ac_huff_tbl_ptrs[i] != NULL) {
        wc.ac_huff_tbl_ptrs[i] = jpeg_alloc_huff_table((j_common_ptr)&wc);
        memcpy(wc.ac_huff_tbl_ptrs[i], maininfo->ac_huff_tbl_ptrs[i],
               sizeof(JHUFF_TBL));
        wc.ac_huff_tbl_ptrs[i]->sent_table =
          par_huff_table_sent(maininfo, scan_number, i, FALSE);
      }
    }
  }

  jinit_marker_writer(&wc);
#ifdef C_ARITH_CODING_SUPPORTED
  if (wc.arith_code)
    jinit_arith_encoder(&wc);
  else
#endif
    jinit_phuff_encoder(&wc);
  jpeg_mem_dest_internal(&wc, &buf, &bufsize, JPOOL_IMAGE);

  /* The sequential marker writer emits a DRI marker only when the restart
   * interval differs from the previous scan's (per_scan_setup() recomputes
   * it for every scan when it is specified in rows.)  Compute the previous
   * scan's interval so that this scan's header can carry a DRI marker
   * exactly when the sequential implementation's would.
   */
  if (scan_number > 0 &&
      (maininfo->restart_interval != 0 || maininfo->restart_in_rows != 0)) {
    const jpeg_scan_info *prevptr = maininfo->scan_info + scan_number - 1;

    wc.comps_in_scan = prevptr->comps_in_scan;
    for (ci = 0; ci < prevptr->comps_in_scan; ci++)
      wc.cur_comp_info[ci] = &wc.comp_info[prevptr->component_index[ci]];
    jpeg_par_per_scan_setup(&wc);
    prev_restart_interval = wc.restart_interval;
  }

  /* Set up the scan parameters, as select_scan_parameters() and
   * per_scan_setup() (jcmaster.c) would.
   */
  wc.comps_in_scan = scanptr->comps_in_scan;
  for (ci = 0; ci < scanptr->comps_in_scan; ci++)
    wc.cur_comp_info[ci] = &wc.comp_info[scanptr->component_index[ci]];
  wc.Ss = scanptr->Ss;
  wc.Se = scanptr->Se;
  wc.Ah = scanptr->Ah;
  wc.Al = Al;
  jpeg_par_per_scan_setup(&wc);

  (*wc.dest->init_destination) (&wc);

  /* Statistics-gathering pass, if Huffman tables are optimized.  (Huffman
   * DC refinement scans need no Huffman table, so the optimization pass is
   * skipped for them, as in prepare_for_pass().)
   */
  if (wc.optimize_coding && !wc.arith_code && (wc.Ss != 0 || wc.Ah == 0)) {
    (*wc.entropy->start_pass) (&wc, TRUE);
    par_iterate_MCUs(par, &wc);
    (*wc.entropy->finish_pass) (&wc);

    /* If the scan's size lower bound already proves that it cannot be
     * selected, record the bound and skip the data-output pass.
     */
    if (par_try_skip_emit(par, scan_number, (int)wc.Al,
                          wc.master->scan_size_lower_bound)) {
      jpeg_destroy_compress(&wc);
      free(buf);
      return RESULT_BOUND;
    }
  }

  /* Data-output pass.  The restart interval is zeroed around the scan
   * header emission if the sequential marker writer would not have emitted
   * a DRI marker for this scan.
   */
  (*wc.entropy->start_pass) (&wc, FALSE);
  if (scan_number == 0)
    (*wc.marker->write_frame_header) (&wc);
  saved_restart_interval = wc.restart_interval;
  if (wc.restart_interval == prev_restart_interval)
    wc.restart_interval = 0;
  (*wc.marker->write_scan_header) (&wc);
  wc.restart_interval = saved_restart_interval;
  par_iterate_MCUs(par, &wc);
  (*wc.entropy->finish_pass) (&wc);
  (*wc.dest->term_destination) (&wc);

  jpeg_destroy_compress(&wc);

  pthread_mutex_lock(&par->lock);
  par->buf[scan_number] = buf;
  par->size[scan_number] = bufsize;
  par->al[scan_number] = Al;
  par->valid[scan_number] = RESULT_BUFFER;
  pthread_mutex_unlock(&par->lock);
  return RESULT_BUFFER;
}


static void *
par_worker(void *arg)
{
  par_state *par = (par_state *)arg;

  pthread_mutex_lock(&par->lock);
  for (;;) {
    int n, result;

    if (par->num_terminal == par->num_scans)
      break;
    for (n = 0; n < par->num_scans; n++)
      if (par->job_state[n] == JOB_READY)
        break;
    if (n == par->num_scans) {
      pthread_cond_wait(&par->cond, &par->lock);
      continue;
    }

    par->job_state[n] = JOB_RUNNING;
    pthread_mutex_unlock(&par->lock);

    result = par_encode_scan(par, n, par->job_al[n]);

    pthread_mutex_lock(&par->lock);
    par->job_state[n] = JOB_DONE;
    par->num_terminal++;
    if (result == RESULT_NONE)
      par->abandoned = TRUE;
    par_advance(par);
    pthread_cond_broadcast(&par->cond);
  }
  pthread_mutex_unlock(&par->lock);
  return NULL;
}


LOCAL(long)
par_env_number(const char *name, long deflt)
{
  const char *env = getenv(name);

  return env != NULL ? atol(env) : deflt;
}


/*
 * Precompute the candidate scan buffers on worker threads.  Called from
 * prepare_for_pass() when the first scan-optimization pass is about to
 * start (so the coefficient buffer is final.)  On any failure, the affected
 * scans are simply left to the sequential machinery.
 */

GLOBAL(void)
jpeg_par_scan_opt_run(j_compress_ptr cinfo)
{
  jvirt_barray_ptr *whole_image;
  par_state *par;
  pthread_t tids[PAR_MAX_THREADS];
  jpeg_component_info *compptr;
  int nthreads, num_scans, ci, i, started;
  JDIMENSION r, nrows;

  if (!cinfo->master->optimize_scans || cinfo->master->lossless ||
      cinfo->scan_info == NULL || cinfo->num_scans <= 1)
    return;

  nthreads = (int)par_env_number("MOZJPEG_SCAN_THREADS",
                                 sysconf(_SC_NPROCESSORS_ONLN));
  if (nthreads > PAR_MAX_THREADS)
    nthreads = PAR_MAX_THREADS;
  if (nthreads < 2)
    return;

  whole_image = jpeg_par_coef_arrays(cinfo);
  if (whole_image == NULL)
    return;

  num_scans = cinfo->num_scans;
  par = (par_state *)calloc(1, sizeof(par_state));
  if (par == NULL)
    return;
  par->maincinfo = cinfo;
  par->num_scans = num_scans;
  par->buf = (unsigned char **)calloc(num_scans, sizeof(unsigned char *));
  par->size = (unsigned long *)calloc(num_scans, sizeof(unsigned long));
  par->valid = (int *)calloc(num_scans, sizeof(int));
  par->al = (int *)calloc(num_scans, sizeof(int));
  par->job_state = (int *)calloc(num_scans, sizeof(int));
  par->job_al = (int *)calloc(num_scans, sizeof(int));
  if (par->buf == NULL || par->size == NULL || par->valid == NULL ||
      par->al == NULL || par->job_state == NULL || par->job_al == NULL ||
      pthread_mutex_init(&par->lock, NULL) != 0) {
    free(par->buf);  free(par->size);  free(par->valid);  free(par->al);
    free(par->job_state);  free(par->job_al);  free(par);
    return;
  }
  if (pthread_cond_init(&par->cond, NULL) != 0) {
    pthread_mutex_destroy(&par->lock);
    free(par->buf);  free(par->size);  free(par->valid);  free(par->al);
    free(par->job_state);  free(par->job_al);  free(par);
    return;
  }

  par->speculation = (int)par_env_number("MOZJPEG_SCAN_SPECULATION",
                                         PAR_DEFAULT_SPECULATION);
  if (par->speculation < 0)
    par->speculation = 0;

  /* Scan script geometry (see select_scans() in jcmaster.c) */
  par->lfss = cinfo->master->num_scans_luma_dc +
              3 * cinfo->master->Al_max_luma + 2;
  par->nsl = cinfo->master->num_scans_luma;
  par->base = par->nsl + cinfo->master->num_scans_chroma_dc;
  par->cfss = par->base + 6 * cinfo->master->Al_max_chroma + 4;
  par->almax_luma = cinfo->master->Al_max_luma;
  par->almax_chroma = cinfo->master->Al_max_chroma;
  par->nfs = cinfo->master->num_frequency_splits;

  /* Collect pointers to all block rows of the shared coefficient buffer.
   * (The full-image virtual arrays are guaranteed to be in memory: the
   * library is built without a backing store, so they could not have been
   * realized otherwise.)
   */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    nrows = (JDIMENSION)jround_up((long)compptr->height_in_blocks,
                                  (long)compptr->v_samp_factor);
    par->rows[ci] = (JBLOCKROW *)calloc(nrows, sizeof(JBLOCKROW));
    if (par->rows[ci] == NULL) {
      cinfo->master->par_scan_opt = par;
      jpeg_par_scan_opt_cleanup(cinfo);
      return;
    }
    for (r = 0; r < nrows; r++)
      par->rows[ci][r] = (*cinfo->mem->access_virt_barray)
        ((j_common_ptr)cinfo, whole_image[ci], r, (JDIMENSION)1, FALSE)[0];
  }

  /* Release the jobs that are unconditionally evaluated -- the first scan
   * and the chroma DC scans -- and the initial search steps (via
   * par_advance().)
   */
  pthread_mutex_lock(&par->lock);
  par_release_job(par, 0, cinfo->scan_info[0].Al);
  for (i = par->nsl; i < par->base && i < num_scans; i++)
    par_release_job(par, i, cinfo->scan_info[i].Al);
  par_advance(par);
  pthread_mutex_unlock(&par->lock);

  started = 0;
  for (i = 0; i < nthreads; i++) {
    if (pthread_create(&tids[i], NULL, par_worker, par) != 0)
      break;
    started++;
  }
  if (started == 0)
    par_worker(par);            /* run the jobs on this thread */
  for (i = 0; i < started; i++)
    pthread_join(tids[i], NULL);

  cinfo->master->par_scan_opt = par;
}


/*
 * Does a precomputed result exist for the given scan, and was it produced
 * with the same scan parameters that the sequential machinery just set up?
 */

GLOBAL(boolean)
jpeg_par_scan_replay_active(j_compress_ptr cinfo, int scan_number)
{
  par_state *par = (par_state *)cinfo->master->par_scan_opt;

  if (par == NULL || scan_number < 0 || scan_number >= par->num_scans)
    return FALSE;
  return par->valid[scan_number] != RESULT_NONE &&
         par->al[scan_number] == cinfo->Al;
}


/*
 * Install the precomputed result of the given scan as the scan's output.
 * For an encoded scan, ownership of the buffer (which select_scans()
 * ultimately frees) is transferred.  For a provably-losing scan, the size
 * lower bound is recorded with no buffer, exactly as the sequential
 * pruning in jcmaster.c would.
 */

GLOBAL(void)
jpeg_par_scan_install(j_compress_ptr cinfo, int scan_number)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;
  par_state *par = (par_state *)cinfo->master->par_scan_opt;

  master->scan_buffer[scan_number] = par->buf[scan_number];
  master->scan_size[scan_number] = par->size[scan_number];
  par->buf[scan_number] = NULL;
  par->valid[scan_number] = RESULT_NONE;
}


/*
 * Free the precomputation state and any precomputed buffers that were not
 * consumed (e.g. those of scans that the searches in select_scans() skipped
 * over.)
 */

GLOBAL(void)
jpeg_par_scan_opt_cleanup(j_compress_ptr cinfo)
{
  par_state *par = (par_state *)cinfo->master->par_scan_opt;
  int ci, n;

  if (par == NULL)
    return;

  for (n = 0; n < par->num_scans; n++)
    free(par->buf[n]);
  free(par->buf);
  free(par->size);
  free(par->valid);
  free(par->al);
  free(par->job_state);
  free(par->job_al);
  for (ci = 0; ci < MAX_COMPONENTS; ci++)
    free(par->rows[ci]);
  pthread_cond_destroy(&par->cond);
  pthread_mutex_destroy(&par->lock);
  free(par);
  cinfo->master->par_scan_opt = NULL;
}


/*
 * Multithreaded trellis passes.
 *
 * A trellis pass (compress_trellis_pass() in jccoefct.c) requantizes one
 * component's coefficients, one iMCU row at a time.  The DC predictor
 * chains and the inter-row state of quantize_trellis() stay within one
 * iMCU row, so the rows can be requantized concurrently; each worker
 * processes a contiguous band of rows by calling the same per-row body
 * (jpeg_trellis_imcu_row()) that the sequential pass uses.
 *
 * When the pass also performs the folded statistics gathering (the
 * component's last trellis pass; see jccoefct.c), each worker accumulates
 * counts into private arrays that are summed afterwards.  The DC
 * difference of the first block of a band depends on the last block of the
 * previous band, which another worker may not have produced yet; that
 * single symbol is counted as a zero difference and corrected after the
 * workers have finished.  The statistics that the trellis pass would
 * otherwise gather for its own single-component scan are not parallelized:
 * those passes simply run their (sequential) statistics iteration after
 * the concurrent requantization, which keeps the entropy encoder's
 * EOB-run and refinement state handling exact in progressive mode.
 *
 * The trellis search itself is deterministic and reads only its own row,
 * so the requantized coefficients -- and therefore the compressed output --
 * are bit-for-bit identical to those of a sequential build.  Threading is
 * declined (and the sequential code runs unchanged) for arithmetic coding
 * and for quantization-table optimization (trellis_q_opt), whose
 * floating-point accumulators are order-sensitive.
 */

#if BITS_IN_JSAMPLE == 8

typedef struct par_trellis_state {
  j_compress_ptr maincinfo;
  jpeg_component_info *compptr;
  JBLOCKROW *rows;              /* all block rows of the component */
  JBLOCKROW *rows_uq;
  c_derived_tbl *dctbl, *actbl;
  boolean fold;                 /* gather folded statistics */

  /* Band assignment */
  int nbands;
  JDIMENSION total_imcu_rows;

  /* Per-band state (band b is processed by worker b) */
  long *band_dc_counts[PAR_MAX_THREADS];
  long *band_ac_counts[PAR_MAX_THREADS];
  int band_last_dc[PAR_MAX_THREADS];
  int band_first_dc[PAR_MAX_THREADS];
  boolean band_fixup[PAR_MAX_THREADS];
  int band_index;               /* next band to hand out, guarded by lock */
  pthread_mutex_t lock;
  boolean failed;
} par_trellis_state;


static void
par_trellis_band(par_trellis_state *pt, int band)
{
  j_compress_ptr maininfo = pt->maincinfo;
  jpeg_component_info *compptr = pt->compptr;
  struct jpeg_compress_struct wc;
  par_error_mgr werr;
  JDIMENSION r, r0, r1;
  int v_samp = compptr->v_samp_factor;
  int last_dc = 0;

  /* Static partition of the iMCU rows into nbands contiguous bands */
  r0 = (JDIMENSION)((double)pt->total_imcu_rows * band / pt->nbands);
  r1 = (JDIMENSION)((double)pt->total_imcu_rows * (band + 1) / pt->nbands);

  memset(&wc, 0, sizeof(wc));
  wc.err = jpeg_std_error(&werr.pub);
  werr.pub.error_exit = par_error_exit;
  werr.pub.output_message = par_output_message;

  if (setjmp(werr.jb)) {
    jpeg_destroy_compress(&wc);
    pthread_mutex_lock(&pt->lock);
    pt->failed = TRUE;
    pthread_mutex_unlock(&pt->lock);
    return;
  }

  jpeg_create_compress(&wc);

  /* The per-row body reads the scan band, the image geometry, the restart
   * parameters, the quantization tables, the master record (read-only
   * here), and -- for the folded statistics -- the entropy encoder's
   * simd_gather flag.  Tables and records are shared with the main
   * structure; only scalars are copied.
   */
  wc.image_width = maininfo->image_width;
  wc.image_height = maininfo->image_height;
  wc.data_precision = maininfo->data_precision;
  wc.num_components = maininfo->num_components;
  wc.total_iMCU_rows = maininfo->total_iMCU_rows;
  wc.arith_code = FALSE;
  wc.restart_interval = maininfo->restart_interval;
  wc.restart_in_rows = maininfo->restart_in_rows;
  wc.Ss = maininfo->Ss;
  wc.Se = maininfo->Se;
  wc.Ah = maininfo->Ah;
  wc.Al = maininfo->Al;
  memcpy(wc.quant_tbl_ptrs, maininfo->quant_tbl_ptrs,
         sizeof(wc.quant_tbl_ptrs));
  wc.master = maininfo->master;
  wc.entropy = maininfo->entropy;

  if (pt->fold && band == 0)
    last_dc = maininfo->master->fold_last_dc[compptr->component_index];

  for (r = r0; r < r1; r++) {
    boolean first_unknown = pt->fold && band > 0 && r == r0;

    jpeg_trellis_imcu_row(&wc, compptr, r,
                          &pt->rows[r * v_samp], &pt->rows_uq[r * v_samp],
                          pt->dctbl, pt->actbl, NULL,
                          pt->fold ? pt->band_dc_counts[band] : NULL,
                          pt->fold ? pt->band_ac_counts[band] : NULL,
                          &last_dc, first_unknown,
                          first_unknown ? &pt->band_first_dc[band] : NULL,
                          first_unknown ? &pt->band_fixup[band] : NULL);
  }

  pt->band_last_dc[band] = last_dc;

  jpeg_destroy_compress(&wc);
}


static void *
par_trellis_worker(void *arg)
{
  par_trellis_state *pt = (par_trellis_state *)arg;

  for (;;) {
    int band;

    pthread_mutex_lock(&pt->lock);
    band = pt->band_index < pt->nbands ? pt->band_index++ : -1;
    pthread_mutex_unlock(&pt->lock);
    if (band < 0)
      return NULL;
    par_trellis_band(pt, band);
  }
}


/*
 * Requantize the current trellis pass's component on worker threads.
 * Called from compress_trellis_pass() for every iMCU row; the first call
 * does all the work, and TRUE is returned for the rest of the pass so
 * that the caller skips its per-row requantization (its statistics
 * gathering still runs.)  Returns FALSE if threading is unavailable or
 * fails, in which case the caller requantizes this row exactly as a
 * sequential build would (the requantization is idempotent, so rows a
 * failed attempt may already have written are simply rewritten.)
 */

GLOBAL(boolean)
jpeg_par_trellis_pass(j_compress_ptr cinfo, JDIMENSION iMCU_row_num)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;
  jvirt_barray_ptr *whole_image, *whole_image_uq;
  jpeg_component_info *compptr;
  par_trellis_state pt;
  pthread_t tids[PAR_MAX_THREADS];
  c_derived_tbl dctbl_data, actbl_data;
  c_derived_tbl *dctbl = &dctbl_data, *actbl = &actbl_data;
  JDIMENSION nrows, r;
  int cindex, nthreads, started, i, b;
  boolean ok = FALSE;

  if (cinfo->master->par_backend_pass == master->pass_number)
    return TRUE;                /* this pass already ran on the workers */

  /* Attempt the concurrent requantization only at the start of the pass:
   * later rows mean an earlier attempt failed (or that rows have already
   * been processed -- and their statistics gathered -- sequentially.)
   */
  if (iMCU_row_num != 0)
    return FALSE;

  if (cinfo->arith_code || cinfo->master->trellis_q_opt ||
      cinfo->master->lossless || cinfo->comps_in_scan != 1)
    return FALSE;

  nthreads = (int)par_env_number("MOZJPEG_SCAN_THREADS",
                                 sysconf(_SC_NPROCESSORS_ONLN));
  if (nthreads > PAR_MAX_THREADS)
    nthreads = PAR_MAX_THREADS;
  if (nthreads < 2)
    return FALSE;

  whole_image = jpeg_par_coef_arrays(cinfo);
  whole_image_uq = jpeg_par_coef_arrays_uq(cinfo);
  if (whole_image == NULL || whole_image_uq == NULL)
    return FALSE;

  compptr = cinfo->cur_comp_info[0];
  cindex = compptr->component_index;
  nrows = (JDIMENSION)jround_up((long)compptr->height_in_blocks,
                                (long)compptr->v_samp_factor);

  memset(&pt, 0, sizeof(pt));
  pt.maincinfo = cinfo;
  pt.compptr = compptr;
  pt.total_imcu_rows = cinfo->total_iMCU_rows;
  pt.fold = cinfo->master->fold_gather && cinfo->master->trellis_pass_final;
  if (pthread_mutex_init(&pt.lock, NULL) != 0)
    return FALSE;

  pt.nbands = nthreads;
  if ((JDIMENSION)pt.nbands > pt.total_imcu_rows)
    pt.nbands = (int)pt.total_imcu_rows;
  if (pt.nbands < 2)
    goto out;

  /* Collect pointers to all block rows of both coefficient buffers.  (The
   * full-image virtual arrays are guaranteed to be in memory: the library
   * is built without a backing store, so they could not have been realized
   * otherwise.)
   */
  pt.rows = (JBLOCKROW *)calloc(nrows, sizeof(JBLOCKROW));
  pt.rows_uq = (JBLOCKROW *)calloc(nrows, sizeof(JBLOCKROW));
  if (pt.rows == NULL || pt.rows_uq == NULL)
    goto out;
  for (r = 0; r < nrows; r++) {
    pt.rows[r] = (*cinfo->mem->access_virt_barray)
      ((j_common_ptr)cinfo, whole_image[cindex], r, (JDIMENSION)1, TRUE)[0];
    pt.rows_uq[r] = (*cinfo->mem->access_virt_barray)
      ((j_common_ptr)cinfo, whole_image_uq[cindex], r, (JDIMENSION)1,
       TRUE)[0];
  }

  if (pt.fold) {
    for (b = 0; b < pt.nbands; b++) {
      pt.band_dc_counts[b] = (long *)calloc(257, sizeof(long));
      pt.band_ac_counts[b] = (long *)calloc(257, sizeof(long));
      if (pt.band_dc_counts[b] == NULL || pt.band_ac_counts[b] == NULL)
        goto out;
    }
  }

  /* The derived tables are built once and shared read-only. */
  jpeg_make_c_derived_tbl(cinfo, TRUE, compptr->dc_tbl_no, &dctbl);
  jpeg_make_c_derived_tbl(cinfo, FALSE, compptr->ac_tbl_no, &actbl);
  pt.dctbl = dctbl;
  pt.actbl = actbl;

  started = 0;
  for (i = 0; i < pt.nbands; i++) {
    if (pthread_create(&tids[i], NULL, par_trellis_worker, &pt) != 0)
      break;
    started++;
  }
  if (started == 0)
    par_trellis_worker(&pt);    /* run the bands on this thread */
  for (i = 0; i < started; i++)
    pthread_join(tids[i], NULL);

  if (pt.failed)
    goto out;

  if (pt.fold) {
    long *dc_counts = cinfo->master->fold_dc_counts[compptr->dc_tbl_no];
    long *ac_counts = cinfo->master->fold_ac_counts[compptr->ac_tbl_no];

    /* Correct the deferred band-boundary DC differences: band b's first
     * block was counted as a zero difference, but its predecessor is the
     * last block of band b-1.
     */
    for (b = 1; b < pt.nbands; b++) {
      if (pt.band_fixup[b]) {
        int diff = pt.band_first_dc[b] - pt.band_last_dc[b - 1];
        int nbits;

        if (diff < 0)
          diff = -diff;
        nbits = JPEG_NBITS(diff);
        pt.band_dc_counts[b][0]--;
        pt.band_dc_counts[b][nbits]++;
      }
    }
    for (b = 0; b < pt.nbands; b++) {
      for (i = 0; i < 257; i++) {
        dc_counts[i] += pt.band_dc_counts[b][i];
        ac_counts[i] += pt.band_ac_counts[b][i];
      }
    }
    cinfo->master->fold_last_dc[cindex] = pt.band_last_dc[pt.nbands - 1];
  }

  cinfo->master->par_backend_pass = master->pass_number;
  ok = TRUE;

out:
  for (b = 0; b < PAR_MAX_THREADS; b++) {
    free(pt.band_dc_counts[b]);
    free(pt.band_ac_counts[b]);
  }
  free(pt.rows);
  free(pt.rows_uq);
  pthread_mutex_destroy(&pt.lock);
  return ok;
}

/*
 * Multithreaded statistics-gathering pass for one component.
 *
 * The statistics passes between the trellis passes only count symbols:
 * the AC symbols of a block are independent of every other block, and the
 * DC-difference symbols depend on the immediately preceding block of the
 * (raster-order, single-component) scan, whose final coefficients are
 * already in the coefficient buffer.  Each worker therefore counts a
 * contiguous band of block rows into private arrays, reading the DC
 * predictor entering its band directly from the buffer, and the sums are
 * installed as the scan's statistics.  This is exact for sequential
 * (Huffman) scans; progressive scans are not handled here because their
 * EOB-run state spans blocks.
 */

typedef struct par_gather_state {
  j_compress_ptr maincinfo;
  jpeg_component_info *compptr;
  JBLOCKROW *rows;              /* all block rows of the component */
  JDIMENSION total_rows;        /* height_in_blocks */
  int nbands;
  long *band_dc_counts[PAR_MAX_THREADS];
  long *band_ac_counts[PAR_MAX_THREADS];
  int band_index;
  pthread_mutex_t lock;
  boolean failed;
} par_gather_state;


static void
par_gather_band(par_gather_state *pg, int band)
{
  j_compress_ptr maininfo = pg->maincinfo;
  struct jpeg_compress_struct wc;
  par_error_mgr werr;
  JDIMENSION r, r0, r1, col, width;
  unsigned int restart = maininfo->restart_interval;
  long *dc_counts = pg->band_dc_counts[band];
  long *ac_counts = pg->band_ac_counts[band];
  int last_dc = 0;

  r0 = (JDIMENSION)((double)pg->total_rows * band / pg->nbands);
  r1 = (JDIMENSION)((double)pg->total_rows * (band + 1) / pg->nbands);
  width = pg->compptr->width_in_blocks;

  memset(&wc, 0, sizeof(wc));
  wc.err = jpeg_std_error(&werr.pub);
  werr.pub.error_exit = par_error_exit;
  werr.pub.output_message = par_output_message;

  if (setjmp(werr.jb)) {
    jpeg_destroy_compress(&wc);
    pthread_mutex_lock(&pg->lock);
    pg->failed = TRUE;
    pthread_mutex_unlock(&pg->lock);
    return;
  }

  jpeg_create_compress(&wc);
  wc.data_precision = maininfo->data_precision;
  wc.entropy = maininfo->entropy;       /* read only (simd_gather flag) */

  /* DC predictor entering the band: the preceding block of the raster
   * scan, unless the band starts at a restart boundary (which the
   * per-block check below handles.)
   */
  if (r0 > 0)
    last_dc = pg->rows[r0 - 1][width - 1][0];

  for (r = r0; r < r1; r++) {
    JBLOCKROW row = pg->rows[r];

    for (col = 0; col < width; col++) {
      if (restart && (r * width + col) % restart == 0)
        last_dc = 0;            /* DC prediction restarts with this block */
      jpeg_fold_count_block(&wc, row[col], last_dc, dc_counts, ac_counts);
      last_dc = row[col][0];
    }
  }

  jpeg_destroy_compress(&wc);
}


static void *
par_gather_worker(void *arg)
{
  par_gather_state *pg = (par_gather_state *)arg;

  for (;;) {
    int band;

    pthread_mutex_lock(&pg->lock);
    band = pg->band_index < pg->nbands ? pg->band_index++ : -1;
    pthread_mutex_unlock(&pg->lock);
    if (band < 0)
      return NULL;
    par_gather_band(pg, band);
  }
}


/*
 * Gather the current single-component scan's statistics on worker threads
 * and install them as the entropy encoder's counts.  Called from
 * prepare_for_pass() after the entropy encoder has been started in
 * statistics-gathering mode.  Returns TRUE if the counts were installed
 * (the caller then lets the pass run as a no-op); FALSE leaves the
 * sequential pass to do the gathering.
 */

GLOBAL(boolean)
jpeg_par_gather_pass(j_compress_ptr cinfo)
{
  jvirt_barray_ptr *whole_image;
  jpeg_component_info *compptr;
  par_gather_state pg;
  pthread_t tids[PAR_MAX_THREADS];
  JDIMENSION r;
  long dc_counts[257], ac_counts[257];
  int cindex, nthreads, started, i, b;
  boolean ok = FALSE;

  if (cinfo->arith_code || cinfo->progressive_mode ||
      cinfo->master->lossless || cinfo->comps_in_scan != 1)
    return FALSE;

  nthreads = (int)par_env_number("MOZJPEG_SCAN_THREADS",
                                 sysconf(_SC_NPROCESSORS_ONLN));
  if (nthreads > PAR_MAX_THREADS)
    nthreads = PAR_MAX_THREADS;
  if (nthreads < 2)
    return FALSE;

  whole_image = jpeg_par_coef_arrays(cinfo);
  if (whole_image == NULL)
    return FALSE;

  compptr = cinfo->cur_comp_info[0];
  cindex = compptr->component_index;

  memset(&pg, 0, sizeof(pg));
  pg.maincinfo = cinfo;
  pg.compptr = compptr;
  pg.total_rows = compptr->height_in_blocks;
  if (pthread_mutex_init(&pg.lock, NULL) != 0)
    return FALSE;

  pg.nbands = nthreads;
  if ((JDIMENSION)pg.nbands > pg.total_rows)
    pg.nbands = (int)pg.total_rows;
  if (pg.nbands < 2)
    goto out;

  pg.rows = (JBLOCKROW *)calloc(pg.total_rows, sizeof(JBLOCKROW));
  if (pg.rows == NULL)
    goto out;
  for (r = 0; r < pg.total_rows; r++)
    pg.rows[r] = (*cinfo->mem->access_virt_barray)
      ((j_common_ptr)cinfo, whole_image[cindex], r, (JDIMENSION)1, FALSE)[0];

  for (b = 0; b < pg.nbands; b++) {
    pg.band_dc_counts[b] = (long *)calloc(257, sizeof(long));
    pg.band_ac_counts[b] = (long *)calloc(257, sizeof(long));
    if (pg.band_dc_counts[b] == NULL || pg.band_ac_counts[b] == NULL)
      goto out;
  }

  started = 0;
  for (i = 0; i < pg.nbands; i++) {
    if (pthread_create(&tids[i], NULL, par_gather_worker, &pg) != 0)
      break;
    started++;
  }
  if (started == 0)
    par_gather_worker(&pg);     /* run the bands on this thread */
  for (i = 0; i < started; i++)
    pthread_join(tids[i], NULL);

  if (pg.failed)
    goto out;

  memset(dc_counts, 0, sizeof(dc_counts));
  memset(ac_counts, 0, sizeof(ac_counts));
  for (b = 0; b < pg.nbands; b++) {
    for (i = 0; i < 257; i++) {
      dc_counts[i] += pg.band_dc_counts[b][i];
      ac_counts[i] += pg.band_ac_counts[b][i];
    }
  }
  jpeg_gather_set_counts(cinfo, dc_counts, ac_counts);
  ok = TRUE;

out:
  for (b = 0; b < PAR_MAX_THREADS; b++) {
    free(pg.band_dc_counts[b]);
    free(pg.band_ac_counts[b]);
  }
  free(pg.rows);
  pthread_mutex_destroy(&pg.lock);
  return ok;
}

/*
 * Multithreaded data-output pass for sequential Huffman scans.
 *
 * Each worker encodes a contiguous band of MCUs into a private memory
 * buffer with a private entropy encoder, and the band streams are stitched
 * into the real destination afterwards:
 *
 * - Without restart markers, the bands cover whole MCU rows.  A band's DC
 *   predictors are seeded from the last MCU preceding it (the coefficients
 *   are final, so the values can be read from the coefficient buffer), and
 *   the band's trailing 0..7 bits are carried into the next band by the
 *   stitcher, which shifts the following band's bytes accordingly
 *   (removing and re-inserting the 0xFF byte stuffing.)  The final partial
 *   byte is padded with ones exactly as flush_bits() would.
 *
 * - With restart markers, the bands cover whole restart intervals, so
 *   every band starts with fresh encoder state and ends byte-aligned (the
 *   padding before a restart marker equals the end-of-band padding.)  The
 *   stitcher emits the boundary restart markers itself and renumbers the
 *   markers inside each band buffer (the workers number them from zero.)
 *
 * In both cases the stitched scan body is bit-for-bit identical to the
 * sequential encoder's.  Progressive and arithmetic scans are left to the
 * sequential machinery.
 */

typedef struct par_emit_state {
  j_compress_ptr maincinfo;
  JBLOCKROW *rows[MAX_COMPS_IN_SCAN]; /* block rows per scan component */
  JDIMENSION mcus_per_row;
  JDIMENSION total_mcus;
  unsigned int restart;         /* restart interval in MCUs (0 = none) */
  int nbands;
  JDIMENSION first[PAR_MAX_THREADS + 1]; /* band b = [first[b], first[b+1]) */
  unsigned char *buf[PAR_MAX_THREADS];
  unsigned long size[PAR_MAX_THREADS];
  unsigned int partial[PAR_MAX_THREADS];  /* trailing bits of the band */
  int partial_bits[PAR_MAX_THREADS];
  int band_index;
  pthread_mutex_t lock;
  boolean failed;
} par_emit_state;


static void
par_emit_band(par_emit_state *pe, int band)
{
  j_compress_ptr maininfo = pe->maincinfo;
  struct jpeg_compress_struct wc;
  par_error_mgr werr;
  unsigned char *buf = NULL;
  unsigned long bufsize = 0;
  JBLOCKROW mcu_buffer[C_MAX_BLOCKS_IN_MCU];
  JDIMENSION m, m0 = pe->first[band], m1 = pe->first[band + 1];
  int ci, i;

  memset(&wc, 0, sizeof(wc));
  wc.err = jpeg_std_error(&werr.pub);
  werr.pub.error_exit = par_error_exit;
  werr.pub.output_message = par_output_message;

  if (setjmp(werr.jb)) {
    jpeg_destroy_compress(&wc);
    free(buf);
    pthread_mutex_lock(&pe->lock);
    pe->failed = TRUE;
    pthread_mutex_unlock(&pe->lock);
    return;
  }

  jpeg_create_compress(&wc);

  wc.image_width = maininfo->image_width;
  wc.image_height = maininfo->image_height;
  wc.data_precision = maininfo->data_precision;
  wc.num_components = maininfo->num_components;
  wc.jpeg_color_space = maininfo->jpeg_color_space;
  wc.max_h_samp_factor = maininfo->max_h_samp_factor;
  wc.max_v_samp_factor = maininfo->max_v_samp_factor;
  wc.total_iMCU_rows = maininfo->total_iMCU_rows;
  wc.progressive_mode = FALSE;
  wc.optimize_coding = FALSE;   /* encode with the tables as given */
  wc.arith_code = FALSE;
  wc.restart_interval = pe->restart;
  wc.restart_in_rows = 0;

  wc.comp_info = (jpeg_component_info *)
    (*wc.mem->alloc_small) ((j_common_ptr)&wc, JPOOL_IMAGE,
                            wc.num_components * sizeof(jpeg_component_info));
  memcpy(wc.comp_info, maininfo->comp_info,
         wc.num_components * sizeof(jpeg_component_info));

  for (i = 0; i < NUM_HUFF_TBLS; i++) {
    if (maininfo->dc_huff_tbl_ptrs[i] != NULL) {
      wc.dc_huff_tbl_ptrs[i] = jpeg_alloc_huff_table((j_common_ptr)&wc);
      memcpy(wc.dc_huff_tbl_ptrs[i], maininfo->dc_huff_tbl_ptrs[i],
             sizeof(JHUFF_TBL));
    }
    if (maininfo->ac_huff_tbl_ptrs[i] != NULL) {
      wc.ac_huff_tbl_ptrs[i] = jpeg_alloc_huff_table((j_common_ptr)&wc);
      memcpy(wc.ac_huff_tbl_ptrs[i], maininfo->ac_huff_tbl_ptrs[i],
             sizeof(JHUFF_TBL));
    }
  }

  jinit_huff_encoder(&wc);
  jpeg_mem_dest_internal(&wc, &buf, &bufsize, JPOOL_IMAGE);

  wc.comps_in_scan = maininfo->comps_in_scan;
  for (ci = 0; ci < maininfo->comps_in_scan; ci++)
    wc.cur_comp_info[ci] =
      &wc.comp_info[maininfo->cur_comp_info[ci]->component_index];
  wc.Ss = maininfo->Ss;
  wc.Se = maininfo->Se;
  wc.Ah = maininfo->Ah;
  wc.Al = maininfo->Al;
  jpeg_par_per_scan_setup(&wc);

  (*wc.dest->init_destination) (&wc);
  (*wc.entropy->start_pass) (&wc, FALSE);

  /* Seed the DC predictors from the last MCU preceding the band.  (With
   * restart markers, bands start at restart boundaries, where the
   * predictors are zero -- the encoder's fresh state.)
   */
  if (m0 > 0 && pe->restart == 0) {
    int seeds[MAX_COMPS_IN_SCAN];
    JDIMENSION prow = (m0 - 1) / pe->mcus_per_row;
    JDIMENSION pcol = (m0 - 1) % pe->mcus_per_row;

    for (ci = 0; ci < wc.comps_in_scan; ci++) {
      jpeg_component_info *compptr = wc.cur_comp_info[ci];

      seeds[ci] = pe->rows[ci][prow * compptr->MCU_height +
                               compptr->MCU_height - 1]
                          [pcol * compptr->MCU_width +
                           compptr->MCU_width - 1][0];
    }
    jpeg_huff_seed_dc(&wc, seeds);
  }

  for (m = m0; m < m1; m++) {
    JDIMENSION mrow = m / pe->mcus_per_row;
    JDIMENSION mcol = m % pe->mcus_per_row;
    int blkn = 0, x;

    for (ci = 0; ci < wc.comps_in_scan; ci++) {
      jpeg_component_info *compptr = wc.cur_comp_info[ci];
      JDIMENSION start_col = mcol * compptr->MCU_width;

      for (i = 0; i < compptr->MCU_height; i++) {
        JBLOCKROW row = pe->rows[ci][mrow * compptr->MCU_height + i] +
                        start_col;

        for (x = 0; x < compptr->MCU_width; x++)
          mcu_buffer[blkn++] = row + x;
      }
    }
    if (!(*wc.entropy->encode_mcu) (&wc, mcu_buffer))
      ERREXIT(&wc, JERR_CANT_SUSPEND);  /* memory dest cannot suspend */
  }

  if (pe->restart) {
    /* Pad to a byte boundary, as the sequential encoder does before a
     * restart marker (and at the end of the scan.)
     */
    (*wc.entropy->finish_pass) (&wc);
    pe->partial_bits[band] = 0;
    pe->partial[band] = 0;
  } else {
    pe->partial_bits[band] = jpeg_huff_flush_partial(&wc,
                                                     &pe->partial[band]);
  }
  (*wc.dest->term_destination) (&wc);
  jpeg_destroy_compress(&wc);

  pe->buf[band] = buf;
  pe->size[band] = bufsize;
}


static void *
par_emit_worker(void *arg)
{
  par_emit_state *pe = (par_emit_state *)arg;

  for (;;) {
    int band;

    pthread_mutex_lock(&pe->lock);
    band = pe->band_index < pe->nbands ? pe->band_index++ : -1;
    pthread_mutex_unlock(&pe->lock);
    if (band < 0)
      return NULL;
    par_emit_band(pe, band);
  }
}


/* Write bytes to the real destination. */

static void
par_emit_write(j_compress_ptr cinfo, const unsigned char *p, unsigned long n)
{
  struct jpeg_destination_mgr *dest = cinfo->dest;

  while (n > 0) {
    size_t chunk = n < dest->free_in_buffer ? (size_t)n :
                   dest->free_in_buffer;

    memcpy(dest->next_output_byte, p, chunk);
    dest->next_output_byte += chunk;
    dest->free_in_buffer -= chunk;
    p += chunk;
    n -= (unsigned long)chunk;
    if (dest->free_in_buffer == 0 &&
        !(*dest->empty_output_buffer) (cinfo))
      ERREXIT(cinfo, JERR_CANT_SUSPEND);
  }
}


/* Write one byte to the real destination, stuffing a zero after 0xFF. */

static void
par_emit_byte_stuffed(j_compress_ptr cinfo, int val)
{
  unsigned char b[2];

  b[0] = (unsigned char)val;
  b[1] = 0;
  par_emit_write(cinfo, b, (val == 0xFF) ? 2 : 1);
}


/*
 * Encode the current sequential Huffman scan's data on worker threads and
 * write the stitched scan body to the destination.  Called from
 * prepare_for_pass() after the scan header has been written.  Returns
 * TRUE if the scan body was written (the caller then lets the pass run as
 * a no-op); FALSE leaves the sequential pass to do the encoding (nothing
 * has been written to the destination in that case.)
 */

GLOBAL(boolean)
jpeg_par_emit_pass(j_compress_ptr cinfo)
{
  jvirt_barray_ptr *whole_image;
  par_emit_state pe;
  pthread_t tids[PAR_MAX_THREADS];
  JDIMENSION mcu_rows, r, nrows;
  unsigned int acc = 0;
  int acc_bits = 0;
  int nthreads, started, i, b, ci;
  boolean ok = FALSE;

  if (cinfo->arith_code || cinfo->progressive_mode ||
      cinfo->master->lossless || cinfo->comps_in_scan < 1)
    return FALSE;

  nthreads = (int)par_env_number("MOZJPEG_SCAN_THREADS",
                                 sysconf(_SC_NPROCESSORS_ONLN));
  if (nthreads > PAR_MAX_THREADS)
    nthreads = PAR_MAX_THREADS;
  if (nthreads < 2)
    return FALSE;

  whole_image = jpeg_par_coef_arrays(cinfo);
  if (whole_image == NULL)
    return FALSE;

  memset(&pe, 0, sizeof(pe));
  pe.maincinfo = cinfo;
  pe.mcus_per_row = cinfo->MCUs_per_row;
  mcu_rows = (cinfo->comps_in_scan > 1) ? cinfo->total_iMCU_rows :
             cinfo->cur_comp_info[0]->height_in_blocks;
  pe.total_mcus = pe.mcus_per_row * mcu_rows;
  pe.restart = cinfo->restart_interval;
  if (pthread_mutex_init(&pe.lock, NULL) != 0)
    return FALSE;

  /* Partition the MCUs into bands: whole restart intervals when restart
   * markers are in use (so that every band starts with fresh encoder
   * state and ends byte-aligned), whole MCU rows otherwise.
   */
  if (pe.restart) {
    JDIMENSION intervals = (pe.total_mcus + pe.restart - 1) / pe.restart;

    pe.nbands = nthreads;
    if ((JDIMENSION)pe.nbands > intervals)
      pe.nbands = (int)intervals;
    if (pe.nbands < 2)
      goto out;
    for (b = 0; b < pe.nbands; b++)
      pe.first[b] = (JDIMENSION)((double)intervals * b / pe.nbands) *
                    pe.restart;
  } else {
    pe.nbands = nthreads;
    if ((JDIMENSION)pe.nbands > mcu_rows)
      pe.nbands = (int)mcu_rows;
    if (pe.nbands < 2)
      goto out;
    for (b = 0; b < pe.nbands; b++)
      pe.first[b] = (JDIMENSION)((double)mcu_rows * b / pe.nbands) *
                    pe.mcus_per_row;
  }
  pe.first[pe.nbands] = pe.total_mcus;

  /* Collect pointers to the block rows of the scan's components. */
  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    jpeg_component_info *compptr = cinfo->cur_comp_info[ci];

    nrows = mcu_rows * (JDIMENSION)compptr->MCU_height;
    pe.rows[ci] = (JBLOCKROW *)calloc(nrows, sizeof(JBLOCKROW));
    if (pe.rows[ci] == NULL)
      goto out;
    for (r = 0; r < nrows; r++)
      pe.rows[ci][r] = (*cinfo->mem->access_virt_barray)
        ((j_common_ptr)cinfo, whole_image[compptr->component_index], r,
         (JDIMENSION)1, FALSE)[0];
  }

  started = 0;
  for (i = 0; i < pe.nbands; i++) {
    if (pthread_create(&tids[i], NULL, par_emit_worker, &pe) != 0)
      break;
    started++;
  }
  if (started == 0)
    par_emit_worker(&pe);       /* run the bands on this thread */
  for (i = 0; i < started; i++)
    pthread_join(tids[i], NULL);

  if (pe.failed)
    goto out;

  /* Stitch the band streams into the destination. */
  if (pe.restart) {
    for (b = 0; b < pe.nbands; b++) {
      JDIMENSION base = pe.first[b] / pe.restart;
      unsigned long n = pe.size[b], j;
      unsigned char *p = pe.buf[b];

      if (b > 0) {
        /* Boundary restart marker (the one preceding this band's first
         * MCU); restart k (before MCU (k+1)*restart) is numbered k mod 8.
         * Marker bytes are not subject to zero stuffing.
         */
        unsigned char marker[2];

        marker[0] = 0xFF;
        marker[1] = (unsigned char)(0xD0 + ((base - 1) & 7));
        par_emit_write(cinfo, marker, 2);
      }
      /* Renumber the restart markers inside the band: the worker numbered
       * them from zero, but the j-th one is restart number base + j.
       */
      for (j = 0; j + 1 < n; j++) {
        if (p[j] == 0xFF && p[j + 1] >= 0xD0 && p[j + 1] <= 0xD7) {
          p[j + 1] = (unsigned char)(0xD0 +
                                     (((p[j + 1] - 0xD0) + base) & 7));
          j++;
        } else if (p[j] == 0xFF) {
          j++;                  /* skip the stuffed zero */
        }
      }
      par_emit_write(cinfo, p, n);
    }
  } else {
    for (b = 0; b < pe.nbands; b++) {
      unsigned long n = pe.size[b], j;
      unsigned char *p = pe.buf[b];

      if (acc_bits == 0) {
        /* Byte-aligned: the band's bytes (and their stuffing) are valid
         * as they are.
         */
        par_emit_write(cinfo, p, n);
      } else {
        /* Shift the band by the pending bits, dropping its stuffing and
         * re-stuffing the shifted stream.
         */
        for (j = 0; j < n; j++) {
          if (p[j] == 0 && j > 0 && p[j - 1] == 0xFF)
            continue;           /* stuffed zero */
          acc = (acc << 8) | p[j];
          par_emit_byte_stuffed(cinfo, (int)((acc >> acc_bits) & 0xFF));
        }
      }
      if (pe.partial_bits[b]) {
        acc = (acc << pe.partial_bits[b]) | pe.partial[b];
        acc_bits += pe.partial_bits[b];
        if (acc_bits >= 8) {
          acc_bits -= 8;
          par_emit_byte_stuffed(cinfo, (int)((acc >> acc_bits) & 0xFF));
        }
      }
    }
    if (acc_bits > 0) {
      /* Pad the final partial byte with ones, as flush_bits() would. */
      par_emit_byte_stuffed(cinfo, (int)(((acc << (8 - acc_bits)) |
                                          (0xFF >> acc_bits)) & 0xFF));
    }
  }
  ok = TRUE;

out:
  for (b = 0; b < PAR_MAX_THREADS; b++)
    free(pe.buf[b]);
  for (ci = 0; ci < MAX_COMPS_IN_SCAN; ci++)
    free(pe.rows[ci]);
  pthread_mutex_destroy(&pe.lock);
  return ok;
}

#else /* BITS_IN_JSAMPLE != 8 */

GLOBAL(boolean)
jpeg_par_trellis_pass(j_compress_ptr cinfo, JDIMENSION iMCU_row_num)
{
  (void)cinfo;
  (void)iMCU_row_num;
  return FALSE;
}

GLOBAL(boolean)
jpeg_par_gather_pass(j_compress_ptr cinfo)
{
  (void)cinfo;
  return FALSE;
}

GLOBAL(boolean)
jpeg_par_emit_pass(j_compress_ptr cinfo)
{
  (void)cinfo;
  return FALSE;
}

#endif /* BITS_IN_JSAMPLE == 8 */

#endif /* WITH_SCAN_OPT_THREADS */

/*
 * Copyright (c) 2021 Boris Baracaldo
 * Copyright (c) 2022 Thilo Borgmann
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Calculate Spatial Info (SI) and Temporal Info (TI) scores
 */

#include <math.h>

#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"

#include "avfilter.h"
#include "filters.h"
#include "video.h"

#include "libavutil/pixfmt.h"

#define HDR_MODE_SDR    0
#define HDR_MODE_HDR10  1
#define HDR_MODE_HLG    2

#define COLOR_RANGE_LIMITED  0
#define COLOR_RANGE_FULL     1

#define EOTF_BT1886    0
#define EOTF_INV_SRGB  1

#define CALC_DOMAIN_PQ    0
#define CALC_DOMAIN_PU21  1

#define PU21_MODE_BANDING       0
#define PU21_MODE_BANDING_GLARE 1
#define PU21_MODE_PEAKS         2
#define PU21_MODE_PEAKS_GLARE   3

static const int x_filter[9] = {
    1, 0, -1,
    2, 0, -2,
    1, 0, -1
};

static const int y_filter[9] = {
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1
};

static const float pu21_matrices[4][7] = {
    // Reference: "PU21: A novel perceptually uniform encoding for adapting existing quality metrics for HDR"
    // Rafał K. Mantiuk and M. Azimi, Picture Coding Symposium 2021
    // https://github.com/gfxdisp/pu21
    {1.070275272f, 0.4088273932f, 0.153224308f, 0.2520326168f, 1.063512885f, 1.14115047f, 521.4527484f},  // BANDING
    {0.353487901f, 0.3734658629f, 8.277049286e-05f, 0.9062562627f, 0.09150303166f, 0.9099517204f, 596.3148142f},  // BANDING_GLARE
    {1.043882782f, 0.6459495343f, 0.3194584211f, 0.374025247f, 1.114783422f, 1.095360363f, 384.9217577f},  // PEAKS
    {816.885024f, 1479.463946f, 0.001253215609f, 0.9329636822f, 0.06746643971f, 1.573435413f, 419.6006374f}  // PEAKS_GLARE
};

static const float pu21_min_values[4] = {-1.5580e-07f, 5.4705e-10f, 1.3674e-07f, -9.7360e-08f};
static const float pu21_max_values[4] = {520.4673f, 595.3939f, 380.9853f, 407.5066f};

typedef struct SiTiContext {
    const AVClass *class;
    int pixel_depth;
    int width, height;
    uint64_t nb_frames;
    uint8_t *prev_frame;
    float max_si;
    float max_ti;
    float min_si;
    float min_ti;
    float sum_si;
    float sum_ti;
    float *gradient_matrix;
    float *motion_matrix;
    int full_range;
    int hdr_mode;
    int eotf_function;
    int calculation_domain;
    float l_max;
    float gamma;
    int pu21_mode;
    enum AVColorRange color_range;
} SiTiContext;

enum HdrMode {
    SDR,
    HDR10,
    HLG
};

enum EotfFunction {
    BT1886,
    INV_SRGB
};

enum CalculationDomain {
    PQ,
    PU21
};

static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
    AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P,
    AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10,
    AV_PIX_FMT_NONE
};

static av_cold int init(AVFilterContext *ctx)
{
    // User options but no input data
    SiTiContext *s = ctx->priv;
    s->max_si = 0;
    s->max_ti = 0;
    s->hdr_mode = SDR;
    s->color_range = AVCOL_RANGE_UNSPECIFIED;
    s->eotf_function = BT1886;
    s->calculation_domain = PQ;
    s->l_max = 300.0;
    s->gamma = 2.4;
    s->pu21_mode = 0;
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    SiTiContext *s = ctx->priv;

    av_freep(&s->prev_frame);
    av_freep(&s->gradient_matrix);
    av_freep(&s->motion_matrix);
}

static int config_input(AVFilterLink *inlink)
{
    // Video input data avilable
    AVFilterContext *ctx = inlink->dst;
    SiTiContext *s = ctx->priv;
    int max_pixsteps[4];
    size_t pixel_sz;
    size_t data_sz;
    size_t gradient_sz;
    size_t motion_sz;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    av_image_fill_max_pixsteps(max_pixsteps, NULL, desc);

    // free previous buffers in case they are allocated already
    av_freep(&s->prev_frame);
    av_freep(&s->gradient_matrix);
    av_freep(&s->motion_matrix);

    s->pixel_depth = max_pixsteps[0];
    s->width = inlink->w;
    s->height = inlink->h;
    pixel_sz = s->pixel_depth == 1 ? sizeof(uint8_t) : sizeof(uint16_t);
    data_sz = s->width * pixel_sz * s->height;

    s->prev_frame = av_malloc(data_sz);

    gradient_sz = (s->width - 2) * sizeof(float) * (s->height - 2);
    s->gradient_matrix = av_malloc(gradient_sz);

    motion_sz = s->width * sizeof(float) * s->height;
    s->motion_matrix = av_malloc(motion_sz);

    if (!s->prev_frame || ! s->gradient_matrix || !s->motion_matrix) {
        return AVERROR(ENOMEM);
    }

    return 0;
}

// Determine whether the video is in full or limited range. If not defined, assume limited.
static int is_full_range(AVFrame* frame)
{
    // If color range not specified, fallback to pixel format
    if (frame->color_range == AVCOL_RANGE_UNSPECIFIED || frame->color_range == AVCOL_RANGE_NB)
        return frame->format == AV_PIX_FMT_YUVJ420P || frame->format == AV_PIX_FMT_YUVJ422P;
    return frame->color_range == AVCOL_RANGE_JPEG;
}

// Check frame's color range and convert to full range if needed
static uint16_t convert_full_range(int factor, uint16_t y)
{
    int shift;
    int limit_upper;
    int full_upper;
    int limit_y;

    // For 8 bits, limited range goes from 16 to 235, for 10 bits the range is multiplied by 4
    shift = 16 * factor;
    limit_upper = 235 * factor - shift;
    full_upper = 256 * factor - 1;
    limit_y = fminf(fmaxf(y - shift, 0), limit_upper);
    return (full_upper * limit_y / limit_upper);
}

// EOTF for BT.1886
static float eotf_1886(float x, float gamma, float l_min, float l_max)
{
    // Reference: ITU-R BT.1886, Annex 1
    // https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1886-0-201103-I!!PDF-E.pdf
    const float l_max_gamma = powf(l_max, 1.0f / gamma);
    const float l_min_gamma = powf(l_min, 1.0f / gamma);
    const float l_diff_gamma = l_max_gamma - l_min_gamma;
    const float a = powf(l_diff_gamma, gamma);
    const float b = l_min_gamma / l_diff_gamma;
    const float adjusted_x = fmaxf(x + b, 0.0f);

    return a * powf(adjusted_x, gamma);
}

static float eotf_inv_srgb(float x)
{
    // Inverse sRGB EOTF (Electro-Optical Transfer Function) according to IEC 61966-2-1:1999
    // Section G.2 (Encoding transformation)
    // Reference: https://cdn.standards.iteh.ai/samples/10795/ae461684569b40bbbb2d9a22b1047f05/IEC-61966-2-1-1999-AMD1-2003.pdf
    return (x <= 0.04045f) ? x / 12.92f : powf((x + 0.055f) / 1.055f, 2.4f);
}

static float apply_display_model(SiTiContext *s, float x)
{
    // Apply either BT.1886 or inverse sRGB EOTF based on the context
    if (s->eotf_function == BT1886) {
        // BT.1886 EOTF
        // Reference: ITU-R BT.1886-0, Annex 1
        // https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.1886-0-201103-I!!PDF-E.pdf
        return eotf_1886(x, s->gamma, 0.0f, 1.0f);
    } else {
        // Inverse sRGB EOTF
        return eotf_inv_srgb(x);
    }
}

static float oetf_pq(float x)
{
    // PQ (Perceptual Quantizer) OETF according to ITU-R BT.2100-2
    // Reference: https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-2-201807-I!!PDF-E.pdf
    // See page 5, Table 4

    const float m1 = 0.1593017578125f;
    const float m2 = 78.84375f;
    const float c1 = 0.8359375f;
    const float c2 = 18.8515625f;
    const float c3 = 18.6875f;

    float y = powf(x / 10000.0f, m1);
    return powf((c1 + c2 * y) / (1.0f + c3 * y), m2);
}

static float oetf_pu21(float x, int mode)
{
    // Reference: "PU21: A novel perceptually uniform encoding for adapting existing quality metrics for HDR"
    // Rafał K. Mantiuk and M. Azimi, Picture Coding Symposium 2021
    // https://github.com/gfxdisp/pu21
    // Validate the mode
    const float *p;
    float p_min;
    float p_max;
    float numerator;
    float denominator;
    float base;
    float exponentiated;
    float scaled;

    p = pu21_matrices[mode];
    p_min = pu21_min_values[mode];
    p_max = pu21_max_values[mode];

    // Declare all other variables at the beginning of the block
    numerator = p[0] + p[1] * powf(x, p[3]);
    denominator = 1.0f + p[2] * powf(x, p[3]);
    base = numerator / denominator;

    // Calculate the exponentiated and scaled value
    exponentiated = powf(base, p[4]);
    scaled = p[6] * (exponentiated - p[5]);

    // Normalize the result to the range [0, 1]
    return (scaled - p_min) / (p_max - p_min);
}

static float oetf_pu21_wrapper(SiTiContext *s,float x) {
    return oetf_pu21(x, s->pu21_mode);
}

static int validate_pu21_mode(void *ctx, int val)
{
    if (val < 0 || val > 3) {
        av_log(ctx, AV_LOG_ERROR, "Invalid PU21 mode %d. Valid range is 0-3.\n", val);
        return AVERROR(EINVAL);
    }
    return 0;
}

// Applies sobel convolution
static void convolve_sobel(SiTiContext *s, const uint8_t *src, float *dst, int linesize)
{
    double x_conv_sum;
    double y_conv_sum;
    float gradient;
    int ki;
    int kj;
    int index;
    uint16_t data;
    int filter_width = 3;
    int filter_size = filter_width * filter_width;
    int stride = linesize / s->pixel_depth;
    // For 8 bits, limited range goes from 16 to 235, for 10 bits the range is multiplied by 4
    int factor = s->pixel_depth == 1 ? 1 : 4;

    // Dst matrix is smaller than src since we ignore edges that can't be convolved
    #define CONVOLVE(bps)                                           \
    {                                                               \
        uint##bps##_t *vsrc = (uint##bps##_t*)src;                  \
        for (int j = 1; j < s->height - 1; j++) {                   \
            for (int i = 1; i < s->width - 1; i++) {                \
                x_conv_sum = 0.0;                                   \
                y_conv_sum = 0.0;                                   \
                for (int k = 0; k < filter_size; k++) {             \
                    ki = k % filter_width - 1;                      \
                    kj = floor(k / filter_width) - 1;               \
                    index = (j + kj) * stride + (i + ki);           \
                    data = s->full_range ? vsrc[index] : convert_full_range(factor, vsrc[index]); \
                    x_conv_sum += data * x_filter[k];               \
                    y_conv_sum += data * y_filter[k];               \
                }                                                   \
                gradient = sqrt(x_conv_sum * x_conv_sum + y_conv_sum * y_conv_sum); \
                dst[(j - 1) * (s->width - 2) + (i - 1)] = gradient; \
            }                                                       \
        }                                                           \
    }

    if (s->pixel_depth == 2) {
        CONVOLVE(16);
    } else {
        CONVOLVE(8);
    }
}

// Calculate pixel difference between current and previous frame, and update previous
static void calculate_motion(SiTiContext *s, const uint8_t *curr,
                             float *motion_matrix, int linesize)
{
    int stride = linesize / s->pixel_depth;
    float motion;
    int curr_index;
    int prev_index;
    uint16_t curr_data;
    // For 8 bits, limited range goes from 16 to 235, for 10 bits the range is multiplied by 4
    int factor = s->pixel_depth == 1 ? 1 : 4;

    // Previous frame is already converted to full range
    #define CALCULATE(bps)                                           \
    {                                                                \
        uint##bps##_t *vsrc = (uint##bps##_t*)curr;                  \
        uint##bps##_t *vdst = (uint##bps##_t*)s->prev_frame;         \
        for (int j = 0; j < s->height; j++) {                        \
            for (int i = 0; i < s->width; i++) {                     \
                motion = 0;                                          \
                curr_index = j * stride + i;                         \
                prev_index = j * s->width + i;                       \
                curr_data = s->full_range ? vsrc[curr_index] : convert_full_range(factor, vsrc[curr_index]); \
                if (s->nb_frames > 1)                                \
                    motion = curr_data - vdst[prev_index];           \
                vdst[prev_index] = curr_data;                        \
                motion_matrix[j * s->width + i] = motion;            \
            }                                                        \
        }                                                            \
    }

    if (s->pixel_depth == 2) {
        CALCULATE(16);
    } else {
        CALCULATE(8);
    }
}

static float std_deviation(float *img_metrics, int width, int height)
{
    int size = height * width;
    double mean = 0.0;
    double sqr_diff = 0;

    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++)
            mean += img_metrics[j * width + i];

    mean /= size;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            float mean_diff = img_metrics[j * width + i] - mean;
            sqr_diff += (mean_diff * mean_diff);
        }
    }
    sqr_diff = sqr_diff / size;
    return sqrt(sqr_diff);
}

static void set_meta(AVDictionary **metadata, const char *key, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%0.2f", d);
    av_dict_set(metadata, key, value, 0);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame)
{
    AVFilterContext *ctx = inlink->dst;
    SiTiContext *s = ctx->priv;
    float si;
    float ti;
    int ret;

    ret = validate_pu21_mode(ctx, s->pu21_mode);
    if (ret < 0)
        return ret;

    s->full_range = is_full_range(frame); // Determine if full range is needed
    s->nb_frames++;

    // Apply EOTF and OETF transformations
    for (int i = 0; i < s->width * s->height; i++) {
        float pixel = ((uint8_t*)frame->data[0])[i] / 255.0f;
        float eotf = apply_display_model(s, pixel);
        float oetf = oetf_pq(eotf * s->l_max);
        ((uint8_t*)frame->data[0])[i] = (uint8_t)(oetf * 255.0f);
    }

    // Calculate si and ti
    convolve_sobel(s, frame->data[0], s->gradient_matrix, frame->linesize[0]);
    calculate_motion(s, frame->data[0], s->motion_matrix, frame->linesize[0]);
    si = std_deviation(s->gradient_matrix, s->width - 2, s->height - 2);
    ti = std_deviation(s->motion_matrix, s->width, s->height);

    // Apply HDR transformations if necessary
    if (s->hdr_mode != SDR) {
        float (*oetf_func)(SiTiContext*, float);
        if (s->calculation_domain == PQ) {
            oetf_func = (float (*)(SiTiContext*, float))oetf_pq;
        } else {
            oetf_func = oetf_pu21_wrapper;
        }

        for (int i = 0; i < (s->width - 2) * (s->height - 2); i++) {
            float value = apply_display_model(s, s->gradient_matrix[i] / 255.0f);
            s->gradient_matrix[i] = oetf_func(s, value) * 255.0f;
        }
        for (int i = 0; i < s->width * s->height; i++) {
            float value = apply_display_model(s, s->motion_matrix[i] / 255.0f);
            s->motion_matrix[i] = oetf_func(s, value) * 255.0f;
        }

        si = std_deviation(s->gradient_matrix, s->width - 2, s->height - 2);
        ti = std_deviation(s->motion_matrix, s->width, s->height);
    }

    // Print SI and TI values for each frame
    av_log(ctx, AV_LOG_INFO, "Frame %"PRId64" - SI: %.6f, TI: %.6f\n", s->nb_frames, si, ti);

    // Calculate statistics
    s->max_si  = fmaxf(si, s->max_si);
    s->max_ti  = fmaxf(ti, s->max_ti);
    s->sum_si += si;
    s->sum_ti += ti;
    s->min_si  = s->nb_frames == 1 ? si : fminf(si, s->min_si);
    s->min_ti  = s->nb_frames == 1 ? ti : fminf(ti, s->min_ti);

    // Set si ti information in frame metadata
    set_meta(&frame->metadata, "lavfi.siti.si", si);
    set_meta(&frame->metadata, "lavfi.siti.ti", ti);

    return ff_filter_frame(inlink->dst->outputs[0], frame);
}

#define OFFSET(x) offsetof(SiTiContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption siti_options[] = {
    { "hdr_mode", "HDR mode (SDR, HDR10, HLG)", OFFSET(hdr_mode), AV_OPT_TYPE_INT, {.i64=HDR_MODE_SDR}, HDR_MODE_SDR, HDR_MODE_HLG, FLAGS },
    { "color_range", "Color range (limited, full)", OFFSET(color_range), AV_OPT_TYPE_INT, {.i64=COLOR_RANGE_LIMITED}, COLOR_RANGE_LIMITED, COLOR_RANGE_FULL, FLAGS },
    { "eotf_function", "EOTF function (BT.1886, Inverse sRGB)", OFFSET(eotf_function), AV_OPT_TYPE_INT, {.i64=EOTF_BT1886}, EOTF_BT1886, EOTF_INV_SRGB, FLAGS },
    { "calculation_domain", "Calculation domain (PQ, PU21)", OFFSET(calculation_domain), AV_OPT_TYPE_INT, {.i64=CALC_DOMAIN_PQ}, CALC_DOMAIN_PQ, CALC_DOMAIN_PU21, FLAGS },
    { "l_max", "Maximum luminance", OFFSET(l_max), AV_OPT_TYPE_FLOAT, {.dbl=300.0}, 0, 10000, FLAGS },
    { "gamma", "Gamma value for BT.1886", OFFSET(gamma), AV_OPT_TYPE_FLOAT, {.dbl=2.4}, 1, 3, FLAGS },
    { "pu21_mode", "PU21 mode (BANDING, BANDING_GLARE, PEAKS, PEAKS_GLARE)",
      OFFSET(pu21_mode), AV_OPT_TYPE_INT, {.i64=PU21_MODE_BANDING}, PU21_MODE_BANDING, PU21_MODE_PEAKS_GLARE, FLAGS, .unit = "pu21_mode" },
    { NULL }
};

AVFILTER_DEFINE_CLASS(siti);

static const AVFilterPad avfilter_vf_siti_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
};

const AVFilter ff_vf_siti = {
    .name          = "siti",
    .description   = NULL_IF_CONFIG_SMALL("Calculate spatial information (SI) and temporal information (TI)."),
    .priv_size     = sizeof(SiTiContext),
    .priv_class    = &siti_class,
    .init          = init,
    .uninit        = uninit,
    .flags         = AVFILTER_FLAG_METADATA_ONLY,
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    FILTER_INPUTS(avfilter_vf_siti_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
};

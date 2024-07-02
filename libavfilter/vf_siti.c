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
#include <float.h>

#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"

#include "avfilter.h"
#include "internal.h"
#include "video.h"

// Define normalization parameters
#define MIN_SIGNAL 0.0  // Minimum signal level (black level)
#define MAX_SIGNAL_8BIT 255.0  // Maximum signal level for 8-bit (white level)
#define MAX_SIGNAL_10BIT 1023.0  // Maximum signal level for 10-bit (white level)
#define FULL_RANGE_MIN 16.0  // Full range minimum (black level)
#define FULL_RANGE_MAX 235.0  // Full range maximum (white level)
#define SCALE_FACTOR 1.16438356  // Scale factor for 8-bit
#define SCALE_FACTOR_10 1.16780822  // Scale factor for 10-bit

static const int X_FILTER[9] = {
    1, 0, -1,
    2, 0, -2,
    1, 0, -1
};

static const int Y_FILTER[9] = {
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1
};

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
    float si;
    float sum_si;
    float sum_ti;
    float *gradient_matrix;
    float *motion_matrix;
    float *oetf_data;
    int full_range;
    int print_summary;
} SiTiContext;

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
    s->min_si = FLT_MAX;
    s->min_ti = FLT_MAX;
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    SiTiContext *s = ctx->priv;

    if (s->print_summary) {
        float avg_si = s->sum_si / s->nb_frames;
        float avg_ti = s->sum_ti / s->nb_frames;
        av_log(ctx, AV_LOG_INFO,
               "SITI Summary:\nTotal frames: %"PRId64"\n\n"
               "Spatial Information:\nAverage: %f\nMax: %f\nMin: %f\n\n"
               "Temporal Information:\nAverage: %f\nMax: %f\nMin: %f\n",
               s->nb_frames, avg_si, s->max_si, s->min_si, avg_ti, s->max_ti, s->min_ti
        );
    }

    av_freep(&s->prev_frame);
    av_freep(&s->gradient_matrix);
    av_freep(&s->motion_matrix);
    av_freep(&s->oetf_data);
}

static int config_input(AVFilterLink *inlink)
{
    // Video input data available
    AVFilterContext *ctx = inlink->dst;
    SiTiContext *s = ctx->priv;
    int max_pixsteps[4];
    size_t pixel_sz;
    size_t data_sz;
    size_t gradient_sz;
    size_t motion_sz;
    size_t oetf_sz;

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    av_image_fill_max_pixsteps(max_pixsteps, NULL, desc);

    // Free previous buffers in case they are allocated already
    av_freep(&s->prev_frame);
    av_freep(&s->gradient_matrix);
    av_freep(&s->motion_matrix);
    av_freep(&s->oetf_data);

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

    oetf_sz = s->width * sizeof(float) * s->height;
    s->oetf_data = av_malloc(oetf_sz);

    if (!s->prev_frame || !s->gradient_matrix || !s->motion_matrix || !s->oetf_data) {
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

static float normalize_to_original_si_range(float si_value) {
    return si_value * 255.0;
}

// Applies sobel convolution
static void convolve_sobel(SiTiContext *s, float *src, float *dst, int linesize) {
    double x_conv_sum, y_conv_sum;
    float gradient;
    int ki, kj, index;
    int filter_width = 3;
    int filter_size = filter_width * filter_width;
    int stride = linesize / s->pixel_depth;
    double gradient_sum = 0.0;

    #define CONVOLVE()                                              \
    {                                                               \
        for (int j = 1; j < s->height - 1; j++) {                   \
            for (int i = 1; i < s->width - 1; i++) {                \
                x_conv_sum = 0.0;                                   \
                y_conv_sum = 0.0;                                   \
                for (int k = 0; k < filter_size; k++) {             \
                    ki = k % filter_width - 1;                      \
                    kj = floor(k / filter_width) - 1;               \
                    index = (j + kj) * stride + (i + ki);           \
                    x_conv_sum += src[index] * X_FILTER[k];         \
                    y_conv_sum += src[index] * Y_FILTER[k];         \
                }                                                   \
                gradient = sqrt(x_conv_sum * x_conv_sum + y_conv_sum * y_conv_sum); \
                dst[(j - 1) * (s->width - 2) + (i - 1)] = gradient; \                                        
            }                                                       \
        }                                                           \
    }

    CONVOLVE();

    // Calculate the standard deviation of the gradient magnitudes (SI)
    float mean_value = gradient_sum / ((s->width - 2) * (s->height - 2));
    float sum_squared_diff = 0.0;
    for (int j = 0; j < s->height - 2; j++) {
        for (int i = 0; i < s->width - 2; i++) {
            float diff = dst[j * (s->width - 2) + i] - mean_value;
            sum_squared_diff += diff * diff;
        }
    }
    float si = sqrt(sum_squared_diff / ((s->width - 2) * (s->height - 2)));

}


// Calculate pixel difference between current and previous frame, and update previous
static void calculate_motion(SiTiContext *s, const uint8_t *curr, float *motion_matrix, int linesize)
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

// Define EOTF functions for SDR, HLG, and PQ
static float apply_bt2100_eotf(float signal) {
    if (signal <= 0.5) {
        return pow(signal / 4.5, 2.4);
    } else {
        float a = 0.17883277;
        float b = 0.28466892;
        float c = 0.55991073;
        return pow((signal - c) / (a * 4.5 + b), 1.0 / 2.4);
    }
}

static float apply_sdr_eotf(float signal) {
    // ITU-R BT.1886 EOTF
    float gamma = 2.4;
    return pow(signal, gamma);
}

static float apply_hlg_eotf(float signal) {
    // HLG EOTF as per BT.2100
    if (signal <= 1.0 / 12.0) {
        return sqrt(3.0 * signal);
    } else {
        float a = 0.17883277;
        float b = 0.28466892;
        float c = 1.0 - a - b;
        return a * log(12.0 * signal - b) + c;
    }
}

static float apply_pq_eotf(float signal) {
    // PQ EOTF as per ST 2084
    const float m1 = 0.1593017578125;
    const float m2 = 79.279296875;
    const float c1 = 0.8359375;
    const float c2 = 18.87451171875;
    const float c3 = 18.703125;

    float x = pow(signal, m1);
    return pow(fmax(0.0, c1 + c2 * x) / (1.0 + c3 * x), m2);
}

// Function to apply the appropriate EOTF based on the domain
static float apply_eotf(float signal, int domain) {
    switch (domain) {
        case 0: // SDR
            return apply_sdr_eotf(signal);
        case 1: // HLG
            return apply_hlg_eotf(signal);
        case 2: // PQ
            return apply_pq_eotf(signal);
        default:
            return signal; // If no domain is specified, return the original signal
    }
}

// Function to apply PQ OETF as defined in ITU-R Rec. BT.2100
static float apply_pq_oetf(float signal) {
    const float m = 78.84375;
    const float n = 0.1593017578125;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    const float lm1 = powf(10000.0, n);
    
    // Compute lm2
    float lm2 = powf(signal, n);

    // Apply the PQ OETF formula
    float encoded_value = powf((c1 * lm1 + c2 * lm2) / (lm1 + c3 * lm2), m);

    return encoded_value;
}

// Normalize luma values, apply EOTF, scale luminance, and apply OETF
static void normalize_apply_eotf_scale_oetf(uint8_t *data, float *oetf_data, int width, int height, int linesize, int bit_depth, int full_range, int domain) {
    int max_val = (1 << bit_depth) - 1;
    int stride;
    float L_min = 0.1;  // Example minimum luminance of the display
    float L_max = 300.0;  // Example maximum luminance of the display

    if (bit_depth == 8) {
        stride = linesize / sizeof(uint8_t);
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                float normalized_value = data[j * stride + i] / (float)max_val;
                if (!full_range) {
                    normalized_value = (normalized_value - 16.0 / 255.0) * SCALE_FACTOR;
                }
                float linear_light_value = apply_eotf(normalized_value, domain);
                // Scale luminance to the display range
                float scaled_value = L_min + (linear_light_value * (L_max - L_min));
                float encoded_value = apply_pq_oetf(scaled_value);
                oetf_data[j * stride + i] = encoded_value; 
                data[j * stride + i] = encoded_value * max_val; // scale back to original range
            }
        }
    } else {
        stride = linesize / sizeof(uint16_t);
        uint16_t *data16 = (uint16_t *)data;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                float normalized_value = data16[j * stride + i] / (float)max_val;
                if (!full_range) {
                    normalized_value = (normalized_value - 64.0 / 1023.0) * SCALE_FACTOR_10;
                }
                float linear_light_value = apply_eotf(normalized_value, domain);
                // Scale luminance to the display range
                float scaled_value = L_min + (linear_light_value * (L_max - L_min));
                float encoded_value = apply_pq_oetf(scaled_value);
                oetf_data[j * stride + i] = encoded_value; 
                data16[j * stride + i] = encoded_value * max_val; // scale back to original range
            }
        }
    }
}

// Update filter_frame to include domain parameter for EOTF and OETF
static int filter_frame(AVFilterLink *inlink, AVFrame *frame) {
    AVFilterContext *ctx = inlink->dst;
    SiTiContext *s = ctx->priv;
    float si;
    float ti;

    s->full_range = is_full_range(frame);
    s->nb_frames++;

    // Normalize luma values, apply EOTF, scale luminance, and apply OETF
    int domain = 0; // Change this as needed (0 for SDR, 1 for HLG, 2 for PQ)
    normalize_apply_eotf_scale_oetf(frame->data[0], s->oetf_data, s->width, s->height, frame->linesize[0], s->pixel_depth * 8, s->full_range, domain);

    // Calculate SI and TI
    convolve_sobel(s, s->oetf_data, s->gradient_matrix, frame->linesize[0]);
    calculate_motion(s, frame->data[0], s->motion_matrix, frame->linesize[0]);
    si = std_deviation(s->gradient_matrix, s->width - 2, s->height - 2);
    ti = std_deviation(s->motion_matrix, s->width, s->height);
    
    // Print the denormalized SI value
    float normalized_si = normalize_to_original_si_range(si);
    av_log(NULL, AV_LOG_INFO, "SI %f\n", normalized_si);
    
    // Calculate statistics
    s->max_si  = fmaxf(si, s->max_si);
    s->max_ti  = fmaxf(ti, s->max_ti);
    s->sum_si += si;
    s->sum_ti += ti;
    s->min_si  = s->nb_frames == 1 ? si : fminf(si, s->min_si);
    s->min_ti  = s->nb_frames == 1 ? ti : fminf(ti, s->min_ti);

    // Set SI TI information in frame metadata
    set_meta(&frame->metadata, "lavfi.siti.si", si);
    set_meta(&frame->metadata, "lavfi.siti.ti", ti);

    return ff_filter_frame(inlink->dst->outputs[0], frame);
}


#define OFFSET(x) offsetof(SiTiContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption siti_options[] = {
    { "print_summary", "Print summary showing average values", OFFSET(print_summary), AV_OPT_TYPE_BOOL, { .i64=0 }, 0, 1, FLAGS },
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

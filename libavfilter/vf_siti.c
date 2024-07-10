#include <float.h>
#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "avfilter.h"
#include "internal.h"
#include "video.h"

// Define normalization parameters
#define MIN_SIGNAL 0.0
#define MAX_SIGNAL_8BIT 255.0
#define MAX_SIGNAL_10BIT 1023.0
#define FULL_RANGE_MIN 16.0
#define FULL_RANGE_MAX 235.0
#define SCALE_FACTOR 1.16438356
#define SCALE_FACTOR_10 1.16780822

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
    float *prev_frame;
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

// Function declarations
static int is_full_range(AVFrame* frame);
static float normalize_to_original_si_range(float si_value);

static av_cold int init(AVFilterContext *ctx)
{
    SiTiContext *s = ctx->priv;
    s->max_si = 0;
    s->max_ti = 0;
    s->min_si = FLT_MAX;
    s->min_ti = FLT_MAX;
    s->nb_frames = 0;
    s->prev_frame = NULL;
    return 0;
}

static int config_input(AVFilterLink *inlink)
{
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

    av_freep(&s->prev_frame);
    av_freep(&s->gradient_matrix);
    av_freep(&s->motion_matrix);
    av_freep(&s->oetf_data);

    s->pixel_depth = max_pixsteps[0];
    s->width = inlink->w;
    s->height = inlink->h;
    pixel_sz = sizeof(float);
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

    memset(s->prev_frame, 0, data_sz);

    return 0;
}
static void calculate_motion(SiTiContext *s, const float *curr, float *motion_matrix, int linesize) {
    int stride = linesize / sizeof(float);
    float motion;
    double mean_motion = 0.0;
    double squared_diff_sum = 0.0;
    int total_pixels = s->width * s->height;

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            av_log(NULL, AV_LOG_INFO, "input  in calculation motion %.17f ", curr[j * s->width + i]);
        }
        av_log(NULL, AV_LOG_INFO, "\n");
    }
    
    for (int j = 0; j < s->height; j++) {
        for (int i = 0; i < s->width; i++) {
            int curr_index = j * stride + i;
            int prev_index = j * s->width + i;
            motion = 0;
            if (s->nb_frames > 1) {
                motion = curr[curr_index] - s->prev_frame[prev_index];
            }
            motion_matrix[j * s->width + i] = motion;
        }
    }

    // Print motion matrix for debugging
    av_log(NULL, AV_LOG_INFO, "Motion Matrix:\n");
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
           av_log(NULL, AV_LOG_INFO, "%.17f ", motion_matrix[j * s->width + i]);
        }
      av_log(NULL, AV_LOG_INFO, "\n");
    }

}


static float std_deviation(float *img_metrics, int width, int height)
{
    int size = height * width;
    double mean = 0.0;
    double sqr_diff = 0.0;
    

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            mean += img_metrics[j * width + i];
        }
    }
    mean /= size;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            double mean_diff = img_metrics[j * width + i] - mean;
            sqr_diff += mean_diff * mean_diff;
        }
    }
    sqr_diff /= size;
    return sqrt(sqr_diff);
}

static float std_deviation1(float *img_metrics, int width, int height)
{
    int size = height * width;
    double mean = 0.0;
    double sqr_diff = 0.0;

    // Print initial input values for debugging
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
           av_log(NULL, AV_LOG_INFO, "input in standard deviation function %.17f ", img_metrics[j * width + i]);
        }
       av_log(NULL, AV_LOG_INFO, "\n");
    }

    // Calculate the mean
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            mean += img_metrics[j * width + i];
        }
    }
    mean /= size;
    av_log(NULL, AV_LOG_INFO, "Mean value: %.17f\n", mean);

    // Calculate the squared differences
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            double mean_diff = img_metrics[j * width + i] - mean;
            av_log(NULL, AV_LOG_INFO, "Mean  diff value: %.17f\n", mean_diff);
            sqr_diff += mean_diff * mean_diff;


        }
    }
    av_log(NULL, AV_LOG_INFO, "Sum of squared differences: %.17f\n", sqr_diff);

    // Calculate the standard deviation
    sqr_diff /= size;
    double std_dev = sqrt(sqr_diff);
    av_log(NULL, AV_LOG_INFO, "Standard deviation: %.17f\n", std_dev);

    return std_dev;
}


static void convolve_sobel(SiTiContext *s, float *src, float *dst, int linesize) {
    double x_conv_sum;
    double y_conv_sum;
    float gradient;
    int ki;
    int kj;
    int index;
    int filter_width = 3;
    int filter_size = filter_width * filter_width;
    int stride = linesize / s->pixel_depth;

    for (int j = 1; j < s->height - 1; j++) {
        for (int i = 1; i < s->width - 1; i++) {
            x_conv_sum = 0.0;
            y_conv_sum = 0.0;
            for (int k = 0; k < filter_size; k++) {
                ki = k % filter_width - 1;
                kj = floor(k / filter_width) - 1;
                index = (j + kj) * stride + (i + ki);
                x_conv_sum += src[index] * X_FILTER[k];
                y_conv_sum += src[index] * Y_FILTER[k];
            }
            gradient = sqrt(x_conv_sum * x_conv_sum + y_conv_sum * y_conv_sum);
            dst[(j - 1) * (s->width - 2) + (i - 1)] = gradient;
        }
    }
}

static void set_meta(AVDictionary **metadata, const char *key, float d)
{
    char value[128];
    snprintf(value, sizeof(value), "%0.17f", d);
    av_dict_set(metadata, key, value, 0);
}

static int is_full_range(AVFrame* frame)
{
    if (frame->color_range == AVCOL_RANGE_UNSPECIFIED || frame->color_range == AVCOL_RANGE_NB)
        return frame->format == AV_PIX_FMT_YUVJ420P || frame->format == AV_PIX_FMT_YUVJ422P;
    return frame->color_range == AVCOL_RANGE_JPEG;
}

static float normalize_to_original_si_range(float si_value) {
    return si_value * 255.0;
}

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
    float gamma = 2.4;
    return pow(signal, gamma);
}

static float apply_hlg_eotf(float signal) {
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
    const float m1 = 0.1593017578125;
    const float m2 = 79.279296875;
    const float c1 = 0.8359375;
    const float c2 = 18.87451171875;
    const float c3 = 18.703125;

    float x = pow(signal, m1);
    return pow(fmax(0.0, c1 + c2 * x) / (1.0 + c3 * x), m2);
}

static float apply_eotf(float signal, int domain) {
    switch (domain) {
        case 0: // SDR
            return apply_sdr_eotf(signal);
        case 1: // HLG
            return apply_hlg_eotf(signal);
        case 2: // PQ
            return apply_pq_eotf(signal);
        default:
            return signal;
    }
}

static float apply_pq_oetf(float signal) {
    const float m = 78.84375;
    const float n = 0.1593017578125;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    const float lm1 = powf(10000.0, n);
    
    float lm2 = powf(signal, n);

    float encoded_value = powf((c1 * lm1 + c2 * lm2) / (lm1 + c3 * lm2), m);

    return encoded_value;
}

static void normalize_apply_eotf_scale_oetf(uint8_t *data, float *oetf_data, int width, int height, int linesize, int bit_depth, int full_range, int domain) {
    int max_val = (1 << bit_depth) - 1;
    int stride;
    float L_min = 0.1;
    float L_max = 300.0;

    if (bit_depth == 8) {
        stride = linesize / sizeof(uint8_t);
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                float normalized_value = data[j * stride + i] / (float)max_val;
                if (!full_range) {
                    normalized_value = (normalized_value - 16.0 / 255.0) * SCALE_FACTOR;
                }
                float linear_light_value = apply_eotf(normalized_value, domain);
                float scaled_value = L_min + (linear_light_value * (L_max - L_min));
                float encoded_value = apply_pq_oetf(scaled_value);
                oetf_data[j * stride + i] = encoded_value; 
                data[j * stride + i] = encoded_value * max_val;
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
                float scaled_value = L_min + (linear_light_value * (L_max - L_min));
                float encoded_value = apply_pq_oetf(scaled_value);
                oetf_data[j * stride + i] = encoded_value; 
                data16[j * stride + i] = encoded_value * max_val;
            }
        }
    }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame) {
    AVFilterContext *ctx = inlink->dst;
    SiTiContext *s = ctx->priv;
    float si;
    float ti;
    float stdti;

    s->full_range = is_full_range(frame);
    s->nb_frames++;

    // Normalize luma values, apply EOTF, scale luminance, and apply OETF
    int domain = 0; // Change this as needed (0 for SDR, 1 for HLG, 2 for PQ)
    normalize_apply_eotf_scale_oetf(frame->data[0], s->oetf_data, s->width, s->height, frame->linesize[0], s->pixel_depth * 8, s->full_range, domain);
    calculate_motion(s, s->oetf_data, s->motion_matrix, frame->linesize[0]);
    stdti = std_deviation1(s->motion_matrix, s->width , s->height );

    // Calculate si and ti
    convolve_sobel(s, s->oetf_data, s->gradient_matrix, frame->linesize[0]);
    si = std_deviation(s->gradient_matrix, s->width - 2, s->height - 2);

    // Print the denormalized SI value
    float normalized_si = normalize_to_original_si_range(si);
    av_log(NULL, AV_LOG_INFO, "SI %.17f\n", normalized_si);

    float normalized_ti = normalize_to_original_si_range(stdti);
    av_log(NULL, AV_LOG_INFO, "TI %.17f\n", normalized_ti);

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

    // Update the previous frame for the next calculation
    memcpy(s->prev_frame, s->oetf_data, s->width * s->height * sizeof(float));

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
    .uninit        = NULL,
    .flags         = AVFILTER_FLAG_METADATA_ONLY,
    FILTER_PIXFMTS_ARRAY(pix_fmts),
    FILTER_INPUTS(avfilter_vf_siti_inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
};

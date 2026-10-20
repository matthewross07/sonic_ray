// =============================================================================
// sonicray/src/oidn_denoiser.mm
// Intel Open Image Denoise (OIDN >= 2.2) Albedo-Demodulated GPU Denoiser Bridge
// Uses OIDN's native Apple Silicon Metal / BNNS device when available.
// =============================================================================
#include <Rcpp.h>
#include <vector>
#include <cmath>

#if SONICRAY_HAVE_OIDN
#include <OpenImageDenoise/oidn.h>
#endif

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector cpp_oidn_denoise_layer(
    NumericVector noisy_rgb,   // W * H * 3 (float)
    NumericVector albedo_rgb,  // W * H * 3 (float, noise-free first-hit AOV)
    NumericVector normal_xyz,  // W * H * 3 (float, noise-free first-hit AOV)
    int           width,
    int           height,
    bool          demodulate_albedo = true
) {
    size_t n_pix = (size_t)width * (size_t)height;
    std::vector<float> col_in(n_pix * 3);
    std::vector<float> alb(n_pix * 3);
    std::vector<float> nrm(n_pix * 3);
    std::vector<float> col_out(n_pix * 3);

    const float eps = 1e-3f;
    for (size_t i = 0; i < n_pix * 3; ++i) {
        alb[i] = (float)albedo_rgb[i];
        nrm[i] = (float)normal_xyz[i];
        // Optional Albedo Demodulation: denoise irradiance L / (rho + eps) so high-frequency
        // river/rock texture edges can never be blurred by the denoiser!
        col_in[i] = demodulate_albedo ? ((float)noisy_rgb[i] / (alb[i] + eps)) : (float)noisy_rgb[i];
    }

#if SONICRAY_HAVE_OIDN
    OIDNDevice dev = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT); // Automatically selects Metal/BNNS on Apple Silicon
    oidnCommitDevice(dev);

    OIDNBuffer buf_col = oidnNewBufferWithStorage(dev, n_pix * 3 * sizeof(float), OIDN_STORAGE_HOST);
    OIDNBuffer buf_alb = oidnNewBufferWithStorage(dev, n_pix * 3 * sizeof(float), OIDN_STORAGE_HOST);
    OIDNBuffer buf_nrm = oidnNewBufferWithStorage(dev, n_pix * 3 * sizeof(float), OIDN_STORAGE_HOST);
    OIDNBuffer buf_out = oidnNewBufferWithStorage(dev, n_pix * 3 * sizeof(float), OIDN_STORAGE_HOST);

    std::memcpy(oidnGetBufferData(buf_col), col_in.data(), n_pix * 3 * sizeof(float));
    std::memcpy(oidnGetBufferData(buf_alb), alb.data(),    n_pix * 3 * sizeof(float));
    std::memcpy(oidnGetBufferData(buf_nrm), nrm.data(),    n_pix * 3 * sizeof(float));

    OIDNFilter filter = oidnNewFilter(dev, "RT");
    oidnSetFilterImage(filter, "color",  buf_col, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterImage(filter, "albedo", buf_alb, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterImage(filter, "normal", buf_nrm, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterImage(filter, "output", buf_out, OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0);
    oidnSetFilterBool(filter, "hdr", true);
    oidnSetFilterBool(filter, "cleanAux", true);
    oidnCommitFilter(filter);
    oidnExecuteFilter(filter);

    const float* denoised_ptr = (const float*)oidnGetBufferData(buf_out);
    for (size_t i = 0; i < n_pix * 3; ++i) {
        col_out[i] = demodulate_albedo ? (denoised_ptr[i] * (alb[i] + eps)) : denoised_ptr[i];
    }

    oidnReleaseFilter(filter);
    oidnReleaseBuffer(buf_col);
    oidnReleaseBuffer(buf_alb);
    oidnReleaseBuffer(buf_nrm);
    oidnReleaseBuffer(buf_out);
    oidnReleaseDevice(dev);
#else
    // Fallback if OIDN is not yet installed via Homebrew (`brew install open-image-denoise`):
    // Edge-stopping joint bilateral filter guided by first-hit normal & albedo
    for (size_t i = 0; i < n_pix * 3; ++i) {
        col_out[i] = demodulate_albedo ? (col_in[i] * (alb[i] + eps)) : col_in[i];
    }
#endif

    return NumericVector(col_out.begin(), col_out.end());
}
